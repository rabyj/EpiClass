"""Main"""
import comet_ml #needed because special snowflake # pylint: disable=unused-import
import pytorch_lightning as pl #in case GCC or CUDA needs it # pylint: disable=unused-import

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import numpy as np
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import PyTorchProfiler, SimpleProfiler
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from argparseutils.directorytype import DirectoryType
from core import metadata
from core import data
from core.model_pytorch import LightningDenseClassifier
from core.trainer import MyTrainer, define_callbacks
from core import analysis
from core.confusion_matrix import ConfusionMatrixWriter


def time_now():
    """Return datetime of call without microseconds"""
    return datetime.utcnow().replace(microsecond=0)


def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("category", type=str, help="The metatada category to analyse.")
    arg_parser.add_argument(
        "hyperparameters", type=Path, help="A json file containing model hyperparameters."
        )
    arg_parser.add_argument("hdf5", type=Path, help="A file with hdf5 filenames. Use absolute path!")
    arg_parser.add_argument("chromsize", type=Path, help="A file with chrom sizes.")
    arg_parser.add_argument("metadata", type=Path, help="A metadata JSON file.")
    arg_parser.add_argument("logdir", type=DirectoryType(), help="A directory for the logs.")
    arg_parser.add_argument("--offline", action="store_true", help="Will log data offline instead of online. Currently cannot merge comet-ml offline outputs.")
    return arg_parser.parse_args(args)

def main(args):
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments(args)

    my_datasource = data.EpiDataSource(
        cli.hdf5,
        cli.chromsize,
        cli.metadata
        )

    with open(cli.hyperparameters, "r", encoding="utf-8") as file:
        hparams = json.load(file)


    # --- Startup LOGGER ---
    #api key in config file
    IsOffline = cli.offline # additional logging fails with True
    exp_name = '-'.join(cli.logdir.parts[-2:])
    comet_logger = pl_loggers.CometLogger(
        project_name="EpiLaP",
        experiment_name=exp_name,
        save_dir=cli.logdir,
        offline=IsOffline,
        auto_metric_logging=False
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
    # my_metadata.merge_molecule_classes()
    # my_metadata.merge_fetal_tissues()


    # --- Dataset selection ---

    # my_metadata = metadata.keep_major_cell_types(my_metadata)
    # my_metadata = metadata.keep_major_cell_types_alt(my_metadata)
    # my_metadata.remove_category_subsets([os.getenv("REMOVE_ASSAY", "")], "assay")
    # my_metadata.select_category_subsets([os.getenv("SELECT_ASSAY", "")], "assay")
    # my_metadata = metadata.special_case_2(my_metadata)

    # my_metadata = metadata.five_cell_types_selection(my_metadata)
    # assays_to_remove = [os.getenv(var, "") for var in ["REMOVE_ASSAY1", "REMOVE_ASSAY2", "REMOVE_ASSAY3"]]
    # my_metadata.remove_category_subsets(assays_to_remove, "assay")

    # epiatlas_assay_list = [
    #     "h3k27ac", "h3k27me3", "h3k36me3", "h3k4me1", "h3k4me3", "h3k9me3",
    #     ]
    # my_metadata.select_category_subsets(epiatlas_assay_list, "assay")


    # --- CREATE training/validation/test SETS (and change metadata according to what is used) ---
    my_data = data.DataSetFactory.from_epidata(
        my_datasource, my_metadata, cli.category, oversample=True, min_class_size=10
        )
    comet_logger.experiment.log_other("Training size", my_data.train.num_examples)
    comet_logger.experiment.log_other("Total nb of files", len(my_metadata))

    print(f"Data+metadata loading time: {time_now() - begin}")

    to_display = set(["assay", cli.category])
    for category in to_display:
        my_metadata.display_labels(category)


    train_dataset = TensorDataset(
        torch.from_numpy(my_data.train.signals).float(),
        torch.from_numpy(np.argmax(my_data.train.labels, axis=-1))
        )

    valid_dataset = TensorDataset(
        torch.from_numpy(my_data.validation.signals).float(),
        torch.from_numpy(np.argmax(my_data.validation.labels, axis=-1))
        )

    train_dataloader = DataLoader(train_dataset, batch_size=hparams.get("batch_size", 64), shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset), pin_memory=True)

    is_training = hparams.get("is_training", True)
    is_tuning = False

    # Warning : output mapping of model created from training dataset
    mapping_file = cli.logdir / "training_mapping.tsv"

    # --- CREATE a brand new MODEL ---
    if is_training and not is_tuning:
        my_data.save_mapping(mapping_file)
        mapping = my_data.load_mapping(mapping_file)
        comet_logger.experiment.log_asset(mapping_file)


        #  DEFINE sizes for input and output LAYERS of the network
        input_size = my_data.train.signals[0].size
        output_size = my_data.train.labels[0].size
        hl_units = int(os.getenv("LAYER_SIZE", default="3000"))
        nb_layers = int(os.getenv("NB_LAYER", default="1"))

        my_model = LightningDenseClassifier(
            input_size=input_size,
            output_size=output_size,
            mapping=mapping,
            hparams=hparams,
            hl_units=hl_units,
            nb_layer=nb_layers
            )

        print("--MODEL STRUCTURE--\n", my_model)
        my_model.print_model_summary()


    # --- RESTORE old model (if just for computing new metrics, or for tuning further) ---
    if not is_training or is_tuning:
        print("No training, loading last best model from given logdir")
        # Note : Training accuracy can vary since reloaded model
        # is not last model (saved when monitored metric does not move anymore)
        my_model = LightningDenseClassifier.restore_model(cli.logdir)


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
                enable_progress_bar=False
                )
        else :
            callbacks.append(pl.callbacks.RichProgressBar(leave=True))
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
        trainer.fit(my_model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)

        trainer.save_model_path()

        training_time = time_now() - before_train
        print(f"training time: {training_time}")

        # reload comet logger for further logging, will create new experience in offline mode
        comet_logger = pl_loggers.CometLogger(
            project_name="EpiLaP",
            save_dir=cli.logdir,
            offline=IsOffline,
            auto_metric_logging=False,
            experiment_key=exp_key
        )
        comet_logger.experiment.log_other("Training time", training_time)
        comet_logger.experiment.log_other("Last epoch", my_model.current_epoch)


    # --- OUTPUTS ---
    my_analyzer = analysis.Analysis(
        my_model, my_data, comet_logger, train_dataset=train_dataset, val_dataset=valid_dataset,
        )

    # --- Print metrics ---
    train_metrics = my_analyzer.get_training_metrics(verbose=True)
    val_metrics = my_analyzer.get_validation_metrics(verbose=True)
    # my_analyzer.get_test_metrics()

    # --- Create prediction file ---
    # my_analyzer.write_training_prediction() # Oversampling = OFF when using this please!
    my_analyzer.write_validation_prediction()
    # my_analyzer.write_test_prediction()


    # --- Create confusion matrix ---
    # my_analyzer.training_confusion_matrix(cli.logdir)
    my_analyzer.validation_confusion_matrix()
    # my_analyzer.test_confusion_matrix(cli.logdir)


    end = time_now()
    main_time = end - begin
    print(f"end {end}")
    print(f"Main() duration: {main_time}")
    comet_logger.experiment.log_other("Main duration", main_time)
    comet_logger.experiment.add_tag("Finished")

if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main(sys.argv[1:])
