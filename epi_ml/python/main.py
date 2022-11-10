"""Main"""
import argparse
import json
import os
import sys
import warnings
from functools import partial
from pathlib import Path

warnings.simplefilter("ignore", category=FutureWarning)

import comet_ml  # needed because special snowflake # pylint: disable=unused-import
import numpy as np
import pytorch_lightning as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
import torch
from pytorch_lightning import callbacks as pl_callbacks
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, TensorDataset

from epi_ml.python.argparseutils.DefaultHelpParser import (
    DefaultHelpParser as ArgumentParser,
)
from epi_ml.python.argparseutils.directorychecker import DirectoryChecker
from epi_ml.python.core import analysis, data, metadata
from epi_ml.python.core.data_source import EpiDataSource
from epi_ml.python.core.model_pytorch import LightningDenseClassifier
from epi_ml.python.core.trainer import MyTrainer, define_callbacks
from epi_ml.python.utils.time import time_now

# pyright: reportUnboundVariable=false


class DatasetError(Exception):
    """Custom error"""

    def __init__(self, *args: object) -> None:
        print(
            "\n--- ERROR : Verify source files, filters, and min_class_size. ---\n",
            file=sys.stderr,
        )
        super().__init__(*args)


def parse_arguments() -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = ArgumentParser()
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
    arg_parser.add_argument(
        "--predict",
        action="store_const",
        const=True,
        help="Enter prediction mode. Will use all data for the test set. Overwrites hparameter file setting. Default mode is training mode.",
    )
    arg_parser.add_argument(
        "--model",
        type=DirectoryChecker(),
        help="Directory from which to load the desired model. Default is logdir.",
    )

    return arg_parser.parse_args()


def main():
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments()

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)

    with open(cli.hyperparameters, "r", encoding="utf-8") as file:
        hparams = json.load(file)

    # # --- Just redo a matrix ---
    # matrix = "test_confusion_matrix"
    # matrix_writer = ConfusionMatrixWriter.from_csv(
    #     csv_path=cli.logdir / f"{matrix}.csv", relative=False
    # )
    # matrix_writer.to_png(cli.logdir / f"{matrix}.png")
    # sys.exit()

    # --- Startup LOGGER ---
    # api key in config file
    IsOffline = cli.offline  # additional logging fails with True
    exp_name = "-".join(cli.logdir.parts[-2:])
    comet_logger = pl_loggers.CometLogger(
        project_name="EpiLaP",
        experiment_name=exp_name,
        save_dir=cli.logdir,
        offline=IsOffline,
        auto_metric_logging=False,
    )
    exp_key = comet_logger.experiment.get_key()
    print(f"The current experiment key is {exp_key}")
    comet_logger.experiment.log_other("Experience key", f"{exp_key}")

    comet_logger.experiment.add_tag(f"{cli.category}")

    if "SLURM_JOB_ID" in os.environ:
        comet_logger.experiment.log_other("SLURM_JOB_ID", os.environ["SLURM_JOB_ID"])
        comet_logger.experiment.add_tag("Cluster")

    # --- LOAD useful info ---
    hdf5_resolution = my_datasource.hdf5_resolution()
    comet_logger.experiment.log_other("HDF5 Resolution", f"{hdf5_resolution/1000}kb")
    # chroms = my_datasource.load_chrom_sizes()

    # --- LOAD DATA ---
    my_metadata = metadata.Metadata(my_datasource.metadata_file)
    assembly = next(iter(my_metadata.datasets)).get("assembly", "NA")
    comet_logger.experiment.add_tag(assembly)
    comet_logger.experiment.log_other("assembly", assembly)

    # --- Categories creation/change ---
    # my_metadata.create_healthy_category()

    # --- Dataset selection ---
    if os.getenv("ASSAY_LIST") is not None:
        assay_list = json.loads(os.environ["ASSAY_LIST"])
        my_metadata.select_category_subsets(cli.category, assay_list)
        print(f"Filtered on {cli.category} to keep {assay_list}")
    else:
        print("No assay list")

    # --- DEFINE current MODE (training, predict or tuning) ---
    is_training = hparams.get("is_training", True)
    is_tuning = False  # HARDCODED FOR THE MOMENT, FINE-TUNNING NOT HANDLED WELL

    if cli.predict is not None:
        is_training = False  # overwrite hparams option
        is_tuning = False
        val_ratio = 0
        test_ratio = 1
        min_class_size = 1
    else:
        val_ratio = 0.1
        test_ratio = 0.1
        min_class_size = 10

    # --- CREATE training/validation/test SETS (and change metadata according to what is used) ---
    time_before_split = time_now()
    oversampling = True
    onehot = False  # current code does not support target onehot encoding anymore

    my_data = data.DataSetFactory.from_epidata(
        my_datasource,
        my_metadata,
        cli.category,
        min_class_size=min_class_size,
        validation_ratio=val_ratio,
        test_ratio=test_ratio,
        onehot=onehot,
        oversample=oversampling,
    )
    print(f"Set loading/splitting time: {time_now() - time_before_split}")

    comet_logger.experiment.log_other("Training size", my_data.train.num_examples)
    comet_logger.experiment.log_other("Total nb of files", len(my_metadata))

    to_display = set(["assay", cli.category])
    for category in to_display:
        my_metadata.display_labels(category)

    train_dataset = None  # the variables all need to exist for the analyzer later
    valid_dataset = None
    test_dataset = None

    # if tuning, all training labels need to be present
    if is_training or is_tuning:

        if my_data.train.num_examples == 0 or my_data.validation.num_examples == 0:
            raise DatasetError("Trying to train without any training or validation data.")

        # transform target labels into int encoding
        if onehot:
            transform = partial(np.argmax, axis=-1)
        else:
            transform = np.array  # already correct encoding

        train_dataset = TensorDataset(
            torch.from_numpy(my_data.train.signals).float(),
            torch.from_numpy(transform(my_data.train.encoded_labels)),
        )

        valid_dataset = TensorDataset(
            torch.from_numpy(my_data.validation.signals).float(),
            torch.from_numpy(transform(my_data.validation.encoded_labels)),
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=hparams.get("batch_size", 64),
            shuffle=True,
            pin_memory=True,
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=len(valid_dataset), pin_memory=True
        )

    # Warning : output mapping of model created from training dataset
    mapping_file = cli.logdir / "training_mapping.tsv"

    # --- CREATE a brand new MODEL ---
    if is_training and not is_tuning:

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

        print("--MODEL STRUCTURE--\n", my_model)
        my_model.print_model_summary()

    # --- RESTORE old model (if just for computing new metrics, or for tuning further) ---

    # Note : Training accuracy can vary since reloaded model
    # is not last model (saved when monitored metric does not move anymore)
    # unless the best_checkpoint.list file is modified
    if not is_training or is_tuning:
        print("No training, loading last best model from model flag.")
        model_dir = cli.logdir
        if cli.model is not None:
            model_dir = cli.model
        my_model = LightningDenseClassifier.restore_model(model_dir)

    if cli.predict:

        if my_data.test.num_examples == 0:
            raise DatasetError("Trying to test without any test data.")

        # remap targets index to correct model mapping
        encoder = my_data.get_encoder(mapping=my_model.mapping)

        test_dataset = TensorDataset(
            torch.from_numpy(my_data.test.signals).float(),
            torch.tensor(
                encoder.transform(my_data.test.original_labels), dtype=torch.int
            ),
        )

    # --- TRAIN the model ---
    if is_training:

        callbacks = define_callbacks(early_stop_limit=hparams.get("early_stop_limit", 20))

        before_train = time_now()

        if torch.cuda.device_count():
            trainer = MyTrainer(
                general_log_dir=cli.logdir,
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
                general_log_dir=cli.logdir,
                last_trained_model=my_model,
                max_epochs=hparams.get("training_epochs", 50),
                check_val_every_n_epoch=hparams.get("measure_frequency", 1),
                logger=comet_logger,
                callbacks=callbacks,
                enable_model_summary=False,
                accelerator="cpu",
                devices=1,
            )

        trainer.print_hyperparameters()
        trainer.fit(
            my_model,
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
        )

        trainer.save_model_path()

        training_time = time_now() - before_train
        print(f"training time: {training_time}")

        # reload comet logger for further logging, will create new experience in offline mode
        comet_logger = pl_loggers.CometLogger(
            project_name="EpiLaP",
            save_dir=cli.logdir,
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

    # --- Print metrics ---
    if is_training or is_tuning:
        train_metrics = my_analyzer.get_training_metrics(verbose=True)
        val_metrics = my_analyzer.get_validation_metrics(verbose=True)
    if cli.predict:
        test_metrics = my_analyzer.get_test_metrics()

    # --- Create prediction file ---
    if is_training or is_tuning:
        # my_analyzer.write_training_prediction() # Oversampling = OFF when using this please!
        my_analyzer.write_validation_prediction()
    if cli.predict:
        my_analyzer.write_test_prediction()

    # --- Create confusion matrix ---
    if is_training or is_tuning:
        my_analyzer.train_confusion_matrix()
        my_analyzer.validation_confusion_matrix()
    if cli.predict:
        my_analyzer.test_confusion_matrix()

    end = time_now()
    main_time = end - begin
    print(f"end {end}")
    print(f"Main() duration: {main_time}")
    comet_logger.experiment.log_other("Main duration", main_time)
    comet_logger.experiment.add_tag("Finished")


if __name__ == "__main__":
    main()
