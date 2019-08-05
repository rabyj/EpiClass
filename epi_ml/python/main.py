import argparse
import datetime
import json
import os
import os.path
import sys
import warnings
warnings.simplefilter("ignore")

import numpy as np

from argparseutils.directorytype import DirectoryType
from core import metadata
from core import data
from core import model
from core import trainer
from core import analysis
from core import visualization

import pickle

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

# @profile
def main(args):
    """main called from command line, edit to change behavior"""
    print('begin {}'.format(datetime.datetime.now()))

    # parse params
    epiml_options = parse_arguments(args)

    # if only want to convert confusion matrix csv to png
    # in_path = os.path.join(epiml_options.logdir, "confusion_matrix.csv")
    # out_path = os.path.join(epiml_options.logdir, "confusion_matrix.png")
    # analysis.convert_matrix_csv_to_png(in_path, out_path)
    # sys.exit()

    # load external files
    my_datasource = data.EpiDataSource(
        epiml_options.hdf5,
        epiml_options.chromsize,
        epiml_options.metadata
        )

    # load useful info
    hdf5_resolution = my_datasource.hdf5_resolution()
    chroms = my_datasource.load_chrom_sizes()

    # load data
    my_metadata = metadata.Metadata.from_epidatasource(my_datasource)
    # my_metadata.select_category_subset(os.getenv("STEP1_ASSAY"), "assay")
    # my_metadata.create_healthy_category()
    # my_metadata.merge_molecule_classes()

    my_data = data.DataSetFactory.from_epidata(
        my_datasource, my_metadata, epiml_options.category, oversample=True, min_class_size=10
        )
    my_metadata.display_labels(epiml_options.category)

    # define sizes for input and output layers of the network
    input_size = my_data.train.signals[0].size
    ouput_size = my_data.train.labels[0].size

    # Assert the resolution is correct so the importance bedgraph works later
    analysis.assert_correct_resolution(chroms, hdf5_resolution, input_size)

    # choose a model
    my_model = model.Dense(input_size, ouput_size)
    #my_model = model.Cnn(41*49, ouput_size, (41, 49))
    #my_model = model.BidirectionalRnn(input_size, ouput_size)

    # trainer for the model
    hparams = json.load(epiml_options.hyperparameters)
    my_trainer = trainer.Trainer(my_data, my_model, epiml_options.logdir, **hparams)

    # train the model
    t0 = datetime.datetime.now()
    my_trainer.train()
    train_time = datetime.datetime.now() - t0
    print('training time: {}'.format(train_time))

    # outputs
    my_analyzer = analysis.Analysis(my_trainer)

    # Print metrics
    my_analyzer.training_metrics()
    my_analyzer.validation_metrics()
    # my_analyzer.test_metrics()

    # Create prediction file
    # outpath1 = os.path.join(epiml_options.logdir, "training_predict.csv")
    outpath2 = os.path.join(epiml_options.logdir, "validation_predict.csv")
    # outpath3 = os.path.join(epiml_options.logdir, "test_predict.csv")

    # my_analyzer.training_prediction(outpath1)
    my_analyzer.validation_prediction(outpath2)
    # my_analyzer.test_prediction(outpath3)

    # Create confusion matrix
    my_analyzer.validation_confusion_matrix(epiml_options.logdir)
    # my_analyzer.test_confusion_matrix(epiml_options.logdir)

    # Create visualisation

    # vis = visualization.Pca()
    # my_trainer.visualize(vis)

    # importance = pickle.load(open("importance.pickle", 'rb'))
    importance = my_analyzer.importance() #TODO: generalize, probably put in model
    # pickle.dump(importance, open("importance.pickle", 'wb'))

    bedgraph_path = os.path.join(epiml_options.logdir, "importance.bedgraph")
    analysis.bedgraph_from_importance(importance, chroms, hdf5_resolution, bedgraph_path)

    print('end {}'.format(datetime.datetime.now()))

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(sys.argv[1:])
