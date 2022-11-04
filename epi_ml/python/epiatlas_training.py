"""Main"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import comet_ml  # needed because special snowflake # pylint: disable=unused-import
import pytorch_lightning as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
import pytorch_lightning.callbacks as pl_callbacks
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, TensorDataset

from epi_ml.python.argparseutils.DefaultHelpParser import (
    DefaultHelpParser as ArgumentParser,
)
from epi_ml.python.argparseutils.directorychecker import DirectoryChecker
from epi_ml.python.core import analysis, metadata
from epi_ml.python.core.data import DataSet
from epi_ml.python.core.data_source import EpiDataSource
from epi_ml.python.core.epiatlas_treatment import EpiAtlasTreatment
from epi_ml.python.core.model_pytorch import LightningDenseClassifier
from epi_ml.python.core.trainer import MyTrainer, define_callbacks
from epi_ml.python.utils.analyze_metadata import (
    filter_cell_types_by_pairs,
    merge_pair_end_info,
)
from epi_ml.python.utils.check_dir import create_dirs
from epi_ml.python.utils.time import time_now


class DatasetError(Exception):
    """Custom error"""

    def __init__(self, *args: object) -> None:
        print(
            "\n--- ERROR : Verify source files, filters, and min_class_size. ---\n",
            file=sys.stderr,
        )
        super().__init__(*args)


def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()

    # fmt: off
    arg_parser.add_argument(
        "category", type=str, help="The metatada category to analyse.",
    )
    arg_parser.add_argument(
        "hyperparameters", type=Path, help="A json file containing model hyperparameters.",
    )
    arg_parser.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!",
    )
    arg_parser.add_argument(
        "chromsize", type=Path, help="A file with chrom sizes.",
    )
    arg_parser.add_argument(
        "metadata", type=Path, help="A metadata JSON file.",
    )
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for the output logs.",
    )
    arg_parser.add_argument(
        "--offline",
        action="store_true",
        help="Will log data offline instead of online. Currently cannot merge comet-ml offline outputs.",
    )
    # fmt: on
    return arg_parser.parse_args(args)


def main(args):
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments(args)

    category = cli.category

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)
    hdf5_resolution = my_datasource.hdf5_resolution()

    with open(cli.hyperparameters, "r", encoding="utf-8") as file:
        hparams = json.load(file)

    my_metadata = metadata.Metadata(my_datasource.metadata_file)

    # --- Prefilter metadata ---
    my_metadata.remove_category_subsets(
        label_category="track_type", labels=["Unique.raw"]
    )

    if category in {"paired", "paired_end_mode"}:
        category = "paired_end_mode"
        merge_pair_end_info(my_metadata)

    label_list = metadata.env_filtering(my_metadata, category)

    if os.getenv("MIN_CLASS_SIZE") is not None:
        min_class_size = int(os.environ["MIN_CLASS_SIZE"])
    else:
        min_class_size = 10

    if category == "harm_sample_ontology_intermediate":
        my_metadata = filter_cell_types_by_pairs(my_metadata)

    # --- Load signals and train ---
    loading_begin = time_now()
    ea_handler = EpiAtlasTreatment(
        my_datasource,
        category,
        label_list,
        n_fold=10,
        test_ratio=0,
        min_class_size=min_class_size,
        metadata=my_metadata,
    )
    loading_time = time_now() - loading_begin

    to_log = {
        "loading_time": str(loading_time),
        "hdf5_resolution": str(hdf5_resolution),
        "category": category,
    }

    time_before_split = time_now()
    for i, my_data in enumerate(ea_handler.yield_split()):

        split_time = time_now() - time_before_split
        to_log.update({"split_time": str(split_time)})

        # --- Startup LOGGER ---
        # api key in config file
        IsOffline = cli.offline  # additional logging fails with True

        logdir = Path(cli.logdir / f"split{i}")
        create_dirs(logdir)

        exp_name = "-".join(cli.logdir.parts[-3:]) + f"-split{i}"
        comet_logger = pl_loggers.CometLogger(
            project_name="EpiLaP",
            experiment_name=exp_name,
            save_dir=logdir,  # type: ignore
            offline=IsOffline,
            auto_metric_logging=False,
        )

        log_pre_training(logger=comet_logger, step=i, to_log=to_log)

        # Everything happens in there
        do_one_experiment(
            split_nb=i,
            my_data=my_data,
            hparams=hparams,
            logger=comet_logger,
        )

        time_before_split = time_now()


def do_one_experiment(
    split_nb: int,
    my_data: DataSet,
    hparams: Dict,
    logger: pl_loggers.CometLogger,
):
    """Wrapper for convenience"""
    begin_loop = time_now()

    logger.experiment.log_other("Training size", my_data.train.num_examples)
    print(f"Split {split_nb} training size: {my_data.train.num_examples}")

    nb_files = len(set(my_data.train.ids.tolist() + my_data.validation.ids.tolist()))
    logger.experiment.log_other("Total nb of files", nb_files)

    train_dataset = None  # the variables all need to exist for the analyzer later
    valid_dataset = None
    test_dataset = None

    if my_data.train.num_examples == 0 or my_data.validation.num_examples == 0:
        raise DatasetError("Trying to train without any training or validation data.")

    train_dataset = TensorDataset(
        torch.from_numpy(my_data.train.signals).float(),
        torch.from_numpy(my_data.train.encoded_labels),
    )

    valid_dataset = TensorDataset(
        torch.from_numpy(my_data.validation.signals).float(),
        torch.from_numpy(my_data.validation.encoded_labels),
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.get("batch_size", 64),
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=len(valid_dataset), pin_memory=True
    )

    # Warning : output mapping of model created from training dataset
    mapping_file = Path(logger.save_dir) / "training_mapping.tsv"  # type: ignore

    # --- CREATE a brand new MODEL ---

    # Create mapping (i --> class string) file
    my_data.save_mapping(mapping_file)
    mapping = my_data.load_mapping(mapping_file)
    logger.experiment.log_asset(mapping_file)

    #  DEFINE sizes for input and output LAYERS of the network
    input_size = my_data.train.signals[0].size
    output_size = len(my_data.classes)
    hl_units = int(os.getenv("LAYER_SIZE", default="3000"))
    nb_layers = int(os.getenv("NB_LAYER", default="1"))

    my_model = LightningDenseClassifier(
        input_size=input_size,
        output_size=output_size,
        mapping=mapping,
        hparams=hparams,
        hl_units=hl_units,
        nb_layer=nb_layers,
    )

    if split_nb == 0:
        print("--MODEL STRUCTURE--\n", my_model)
        my_model.print_model_summary()

    # --- TRAIN the model ---
    if split_nb == 0:
        callbacks = define_callbacks(
            early_stop_limit=hparams.get("early_stop_limit", 20), show_summary=True
        )
    else:
        callbacks = define_callbacks(
            early_stop_limit=hparams.get("early_stop_limit", 20), show_summary=False
        )

    # --- TRAIN the model ---
    callbacks = define_callbacks(early_stop_limit=hparams.get("early_stop_limit", 20))

    before_train = time_now()

    if torch.cuda.device_count():
        trainer = MyTrainer(
            general_log_dir=logger.save_dir,  # type: ignore
            last_trained_model=my_model,
            max_epochs=hparams.get("training_epochs", 50),
            check_val_every_n_epoch=hparams.get("measure_frequency", 1),
            logger=logger,
            callbacks=callbacks,
            enable_model_summary=False,
            accelerator="gpu",
            devices=1,
            precision=16,
            enable_progress_bar=False,
        )
    else:
        callbacks.append(pl_callbacks.RichProgressBar(leave=True))
        trainer = MyTrainer(
            general_log_dir=logger.save_dir,  # type: ignore
            last_trained_model=my_model,
            max_epochs=hparams.get("training_epochs", 50),
            check_val_every_n_epoch=hparams.get("measure_frequency", 1),
            logger=logger,
            callbacks=callbacks,
            enable_model_summary=False,
            accelerator="cpu",
            devices=1,
        )

    if split_nb == 0:
        trainer.print_hyperparameters()
        trainer.fit(
            my_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
            verbose=True,
        )
    else:
        trainer.fit(
            my_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
            verbose=False,
        )
    trainer.save_model_path()

    training_time = time_now() - before_train
    print(f"training time: {training_time}")

    # reload comet logger for further logging, will create new experience in offline mode
    if type(logger.experiment).__name__ == "OfflineExperiment":
        IsOffline = True
    else:
        IsOffline = False

    logger = pl_loggers.CometLogger(
        project_name="EpiLaP",
        save_dir=logger.save_dir,  # type: ignore
        offline=IsOffline,
        auto_metric_logging=False,
        experiment_key=logger.experiment.get_key(),
    )
    logger.experiment.log_metric("Training time", training_time, step=split_nb)
    logger.experiment.log_metric("Last epoch", my_model.current_epoch, step=split_nb)

    # --- OUTPUTS ---
    my_analyzer = analysis.Analysis(
        my_model,
        my_data,
        logger,
        train_dataset=train_dataset,
        val_dataset=valid_dataset,
        test_dataset=test_dataset,
    )

    my_analyzer.get_training_metrics(verbose=True)
    my_analyzer.get_validation_metrics(verbose=True)

    my_analyzer.write_validation_prediction()
    my_analyzer.validation_confusion_matrix()

    end_loop = time_now()
    loop_time = end_loop - begin_loop
    logger.experiment.log_metric("Loop time", loop_time, step=split_nb)
    print(f"Loop time (excludes split time): {loop_time}")

    logger.experiment.add_tag("Finished")
    logger.finalize(status="Finished")

    del logger
    del my_analyzer
    del my_model
    del trainer
    del train_dataset
    del valid_dataset
    del train_dataloader
    del valid_dataloader


def log_pre_training(logger: pl_loggers.CometLogger, step: int, to_log: Dict[str, str]):
    """Log a bunch of stuff in comet logger. Return experience key (str).

    to_log needs:
    - category
    - hdf5_resolution
    - loading_time (initial, for hdf5)
    - split_time (generator yield time)
    """
    # General stuff
    logger.experiment.add_tag("EpiAtlas")

    category = to_log["category"]
    logger.experiment.add_tag(category)
    logger.experiment.log_other("category", category)

    if os.getenv("SLURM_JOB_ID") is not None:
        logger.experiment.log_other("SLURM_JOB_ID", os.environ["SLURM_JOB_ID"])
        logger.experiment.add_tag("Cluster")

    logger.experiment.log_other(
        "HDF5 Resolution", f"{int(to_log['hdf5_resolution'])/1000}kb"
    )

    # Code time stuff
    loading_time = to_log["loading_time"]
    print(f"Initial hdf5 loading time: {loading_time}")
    logger.experiment.log_other("Initial hdf5 loading time", loading_time)

    split_time = to_log["split_time"]
    print(f"Set loading/splitting time: {split_time}")
    logger.experiment.log_metric("Split_time", split_time, step=step)

    # exp id
    exp_key = logger.experiment.get_key()
    print(f"The current experiment key is {exp_key}")
    logger.experiment.log_other("Experience key", f"{exp_key}")


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main(sys.argv[1:])
