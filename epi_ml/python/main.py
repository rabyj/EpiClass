import torch #import first because of library linking (cuda) reasons
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import argparse
import datetime
import json
import os
import os.path
import sys
import warnings
warnings.simplefilter("ignore")

from argparseutils.directorytype import DirectoryType
from core import metadata
from core import data
from core import pytorch_model_test
from core.trainer import define_callbacks
# from core import analysis
# from core import visualization

# import pickle

def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('category', type=str, help='The metatada category to analyse.')
    arg_parser.add_argument('hyperparameters', type=argparse.FileType('r'), help='A json file containing model hyperparameters.')
    arg_parser.add_argument('hdf5', type=argparse.FileType('r'), help='A file with hdf5 filenames. Use absolute path!')
    arg_parser.add_argument('chromsize', type=argparse.FileType('r'), help='A file with chrom sizes.')
    arg_parser.add_argument('metadata', type=argparse.FileType('r'), help='A metadata JSON file.')
    arg_parser.add_argument('logdir', type=DirectoryType(), help='A directory for the logs.')
    return arg_parser.parse_args(args)

def main(args):
    """main called from command line, edit to change behavior"""
    begin = datetime.datetime.now()
    print("begin {}".format(begin))

    # --- PARSE params ---
    epiml_options = parse_arguments(args)

    # if only want to convert confusion matrix csv to png
    # in_path = os.path.join(epiml_options.logdir, "confusion_matrix.csv")
    # out_path = os.path.join(epiml_options.logdir, "confusion_matrix.png")
    # analysis.convert_matrix_csv_to_png(in_path, out_path)
    # sys.exit()

    # --- LOAD external files ---
    my_datasource = data.EpiDataSource(
        epiml_options.hdf5,
        epiml_options.chromsize,
        epiml_options.metadata
        )

    hparams = json.load(epiml_options.hyperparameters)

    # --- LOAD useful info ---
    # hdf5_resolution = my_datasource.hdf5_resolution()
    # chroms = my_datasource.load_chrom_sizes()

    # --- LOAD DATA ---
    my_metadata = metadata.Metadata.from_epidatasource(my_datasource)

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

    # --- CREATE training/validation/test SETS (and change metadata according to what is used) ---
    my_data = data.DataSetFactory.from_epidata(
        my_datasource, my_metadata, epiml_options.category, oversample=True, min_class_size=10
        )

    to_display = set(["assay", epiml_options.category])
    for category in to_display:
        my_metadata.display_labels(category)

    train_dataset = TensorDataset(
        torch.from_numpy(my_data.train.signals),
        torch.from_numpy(my_data.train.labels)
        )

    valid_dataset = TensorDataset(
        torch.from_numpy(my_data.validation.signals),
        torch.from_numpy(my_data.validation.labels)
        )

    print(
        torch.from_numpy(my_data.train.signals).dtype,
        torch.from_numpy(my_data.train.labels).dtype,
        torch.from_numpy(my_data.validation.signals).dtype,
        torch.from_numpy(my_data.validation.labels).dtype
    )

    train_dataloader = DataLoader(train_dataset, batch_size=hparams.get("batch_size", 64), shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=len(valid_dataset))

    my_data.save_mapping(os.path.join(epiml_options.logdir, "training_mapping.tsv"))

    # --- DEFINE sizes for input and output LAYERS of the network ---
    input_size = my_data.train.signals[0].size
    output_size = my_data.train.labels[0].size
    # hl_units = int(os.getenv("LAYER_SIZE", default="3000"))
    # nb_layers = int(os.getenv("NB_LAYER", default="1"))

    # --- Assert the resolution is correct so the importance bedgraph works later ---
    # analysis.assert_correct_resolution(chroms, hdf5_resolution, input_size)

    # --- CREATE a brand new MODEL ---
    my_model = pytorch_model_test.LightningDenseClassifier(input_size, output_size, hparams, hl_units=1000, nb_layer=1)

    print("--MODEL STRUCTURE--\n", my_model)
    my_model.print_info_summary()

    # --- DEFINE training CALLBACKS ---
    callbacks = define_callbacks(early_stop_limit=hparams.get("early_stop_limit", 15))
    tb_logger = pl_loggers.TensorBoardLogger(epiml_options.logdir)

    # --- TRAIN the model ---
    before_train = datetime.datetime.now()

    trainer = pl.Trainer(
        max_epochs=hparams.get("training_epochs", 100),
        check_val_every_n_epoch=hparams.get("measure_frequency", 1),
        logger=tb_logger,
        callbacks=callbacks,
        enable_model_summary=False
        )

    trainer.fit(my_model, train_dataloader, valid_dataloader)

    print("training time: {}".format(datetime.datetime.now() - before_train))

    # --- restore old model ---
    # my_trainer.restore()

    # --- outputs ---
    # my_analyzer = analysis.Analysis(my_trainer)

    # --- Print metrics ---
    # my_analyzer.training_metrics()
    # my_analyzer.validation_metrics()
    # my_analyzer.test_metrics()

    # --- Create prediction file ---
    # outpath1 = os.path.join(epiml_options.logdir, "training_predict.csv")
    # outpath2 = os.path.join(epiml_options.logdir, "validation_predict.csv")
    # outpath3 = os.path.join(epiml_options.logdir, "test_predict.csv")

    # my_analyzer.training_prediction(outpath1)
    # my_analyzer.validation_prediction(outpath2)
    # my_analyzer.test_prediction(outpath3)

    # --- Create confusion matrix ---
    # my_analyzer.training_confusion_matrix(epiml_options.logdir)
    # my_analyzer.validation_confusion_matrix(epiml_options.logdir)
    # my_analyzer.test_confusion_matrix(epiml_options.logdir)

    # --- Create visualisation ---

    # vis = visualization.Pca()
    # my_trainer.visualize(vis)

    # --- Compute/write importance ---

    # importance = pickle.load(open("importance.pickle", 'rb'))
    # importance = my_analyzer.importance() #TODO: generalize, probably put in model
    # pickle.dump(importance, open("importance.pickle", 'wb'))

    # bedgraph_path = os.path.join(epiml_options.logdir, "importance.bedgraph")
    # analysis.values_to_bedgraph(importance, chroms, hdf5_resolution, bedgraph_path)

    end = datetime.datetime.now()
    print("end {}".format(end))
    print("Main() time: {}".format(end - begin))

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(sys.argv[1:])
