"""Main"""
import argparse
import json
import os
import sys
import warnings
from pathlib import Path

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import comet_ml  # needed because special snowflake # pylint: disable=unused-import
import pytorch_lightning as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
import pytorch_lightning.callbacks as pl_callbacks
import torch
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, TensorDataset

from epi_ml.python.argparseutils.directorychecker import DirectoryChecker
from epi_ml.python.core import analysis, metadata
from epi_ml.python.core.data_source import EpiDataSource
from epi_ml.python.core.epiatlas_treatment import EpiAtlasTreatment
from epi_ml.python.core.model_pytorch import LightningDenseClassifier
from epi_ml.python.core.trainer import MyTrainer, define_callbacks
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
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "category", type=str, help="The metatada category to analyse."
    )
    arg_parser.add_argument(
        "hyperparameters",
        type=Path,
        help="A json file containing model hyperparameters.",
    )
    arg_parser.add_argument(
        "hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!"
    )
    arg_parser.add_argument("chromsize", type=Path, help="A file with chrom sizes.")
    arg_parser.add_argument("metadata", type=Path, help="A metadata JSON file.")
    arg_parser.add_argument(
        "logdir", type=DirectoryChecker(), help="Directory for the output logs."
    )
    arg_parser.add_argument(
        "--offline",
        action="store_true",
        help="Will log data offline instead of online. Currently cannot merge comet-ml offline outputs.",
    )

    return arg_parser.parse_args(args)


def main(args):
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments(args)

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)

    with open(cli.hyperparameters, "r", encoding="utf-8") as file:
        hparams = json.load(file)

    hdf5_resolution = my_datasource.hdf5_resolution()

    my_metadata = metadata.Metadata(my_datasource.metadata_file)

    # --- Prefilter metadata ---
    my_metadata.remove_category_subsets(
        label_category="track_type", labels=["Unique.raw"]
    )

    if os.getenv("EXCLUDE_LIST") is not None:
        exclude_list = json.loads(os.environ["EXCLUDE_LIST"])
        my_metadata.remove_category_subsets(
            label_category=cli.category, labels=exclude_list
        )

    if os.getenv("ASSAY_LIST") is not None:
        assay_list = json.loads(os.environ["ASSAY_LIST"])
        print(f"Going to only keep targets with {assay_list}")
    else:
        assay_list = my_metadata.unique_classes(cli.category)
        print("No assay list")

    if os.getenv("MIN_CLASS_SIZE") is not None:
        min_class_size = int(os.environ["MIN_CLASS_SIZE"])
    else:
        min_class_size = 10

    # --- Load signals and train ---
    loading_begin = time_now()
    ea_handler = EpiAtlasTreatment(
        my_datasource,
        cli.category,
        assay_list,
        n_fold=10,
        test_ratio=0,
        min_class_size=min_class_size,
    )
    loading_time = time_now() - loading_begin
    print(f"Initial hdf5 loading time: {loading_time}")

    time_before_split = time_now()
    for i, my_data in enumerate(ea_handler.yield_split()):
        iteration_time = time_now() - time_before_split
        print(f"Set loading/splitting time: {iteration_time}")

        begin_loop = time_now()

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
        comet_logger.experiment.log_other("Initial hdf5 loading time", loading_time)

        comet_logger.experiment.log_metric("Split_time", f"{iteration_time}", step=i)

        exp_key = comet_logger.experiment.get_key()
        print(f"The current experiment key is {exp_key}")
        comet_logger.experiment.log_other("Experience key", f"{exp_key}")

        comet_logger.experiment.add_tag(f"{cli.category}")
        comet_logger.experiment.add_tag("EpiAtlas")

        if os.getenv("SLURM_JOB_ID") is not None:
            comet_logger.experiment.log_other(
                "SLURM_JOB_ID", os.environ["SLURM_JOB_ID"]
            )
            comet_logger.experiment.add_tag("Cluster")

        comet_logger.experiment.log_other(
            "HDF5 Resolution", f"{hdf5_resolution/1000}kb"
        )

        comet_logger.experiment.log_other("Training size", my_data.train.num_examples)
        print(f"Split {i} training size: {my_data.train.num_examples}")

        nb_files = len(
            set(my_data.train.ids.tolist() + my_data.validation.ids.tolist())
        )
        comet_logger.experiment.log_other("Total nb of files", nb_files)

        train_dataset = None  # the variables all need to exist for the analyzer later
        valid_dataset = None
        test_dataset = None

        if my_data.train.num_examples == 0 or my_data.validation.num_examples == 0:
            raise DatasetError(
                "Trying to train without any training or validation data."
            )

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
        mapping_file = logdir / "training_mapping.tsv"

        # --- CREATE a brand new MODEL ---

        # Create mapping (i --> class string) file
        my_data.save_mapping(mapping_file)
        mapping = my_data.load_mapping(mapping_file)
        comet_logger.experiment.log_asset(mapping_file)

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

        if i == 0:
            print("--MODEL STRUCTURE--\n", my_model)
            my_model.print_model_summary()

        # --- TRAIN the model ---
        if i == 0:
            callbacks = define_callbacks(
                early_stop_limit=hparams.get("early_stop_limit", 20), show_summary=True
            )
        else:
            callbacks = define_callbacks(
                early_stop_limit=hparams.get("early_stop_limit", 20), show_summary=False
            )

        # --- TRAIN the model ---
        callbacks = define_callbacks(
            early_stop_limit=hparams.get("early_stop_limit", 20)
        )

        before_train = time_now()

        if torch.cuda.device_count():
            trainer = MyTrainer(
                general_log_dir=logdir,  # type: ignore
                last_trained_model=my_model,
                max_epochs=hparams.get("training_epochs", 50),
                check_val_every_n_epoch=hparams.get("measure_frequency", 1),
                logger=comet_logger,
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
                general_log_dir=logdir,  # type: ignore
                last_trained_model=my_model,
                max_epochs=hparams.get("training_epochs", 50),
                check_val_every_n_epoch=hparams.get("measure_frequency", 1),
                logger=comet_logger,
                callbacks=callbacks,
                enable_model_summary=False,
                accelerator="cpu",
                devices=1,
            )

        if i == 0:
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
        comet_logger = pl_loggers.CometLogger(
            project_name="EpiLaP",
            save_dir=logdir,  # type: ignore
            offline=IsOffline,
            auto_metric_logging=False,
            experiment_key=exp_key,
        )
        comet_logger.experiment.log_other("Training time", training_time)
        comet_logger.experiment.log_other("Last epoch", my_model.current_epoch)

        # --- OUTPUTS ---
        my_analyzer = analysis.Analysis(
            my_model,
            my_data,
            comet_logger,
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
        comet_logger.experiment.log_metric("Loop time", loop_time, step=i)
        print(f"Loop time (excludes split time): {loop_time}")

        comet_logger.experiment.add_tag("Finished")
        comet_logger.finalize(status="Finished")
        time_before_split = time_now()

        del comet_logger
        del my_analyzer
        del my_model
        del trainer
        del train_dataset
        del valid_dataset
        del train_dataloader
        del valid_dataloader


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main(sys.argv[1:])
