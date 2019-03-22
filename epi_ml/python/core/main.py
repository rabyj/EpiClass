import os
import sys
import warnings
warnings.simplefilter("ignore")
import os.path
import numpy as np
import datetime

import metadata
import data
import model
import trainer
import analysis
import visualization

import argparse
from argparseutils.directorytype import DirectoryType

def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('category', type=str, help='The metatada category to analyse.')
    arg_parser.add_argument('hdf5', type=argparse.FileType('r'), help='A file with hdf5 filenames. Use absolute path!')
    arg_parser.add_argument('chromsize', type=argparse.FileType('r'), help='A file with chrom sizes.')
    arg_parser.add_argument('metadata', type=argparse.FileType('r'), help='A metadata JSON file.')
    arg_parser.add_argument('logdir', type=DirectoryType(), help='A directory for the logs.')
    return arg_parser.parse_args(args)

def main(args):
    """
    main called from command line, edit to change behavior
    """
    print('begin {}'.format(datetime.datetime.now()))
    #parse params
    epiml_options = parse_arguments(args)

    #if only want to convert confusion matrix csv to png
    # in_path = os.path.join(epiml_options.logdir, "confusion_matrix.csv")
    # out_path = os.path.join(epiml_options.logdir, "confusion_matrix.png")
    # analysis.convert_matrix_csv_to_png(in_path, out_path)
    # sys.exit()

    #load external files
    my_datasource = data.EpiDataSource(
        epiml_options.hdf5,
        epiml_options.chromsize,
        epiml_options.metadata)

    #load data
    my_metadata = metadata.Metadata(my_datasource)
    my_data = data.EpiData(my_datasource, my_metadata, epiml_options.category, oversample=True, min_class_size=10)
    my_metadata.display_labels(epiml_options.category)

    #define sizes for input and output layers of the network
    input_size = my_data.train.signals[0].size
    ouput_size = my_data.train.labels[0].size

    #choose a model
    my_model = model.Dense(input_size, ouput_size)
    #my_model = model.Cnn(41*49, ouput_size, (41, 49))
    #my_model = model.BidirectionalRnn(input_size, ouput_size)

    #trainer for the model
    hparams = {
            "learning_rate": 1e-6,
            "training_epochs": 200,
            "batch_size": 64,
            "measure_frequency": 1,
            "l1_scale": 0.001, #ONLY IN L1DENSE
            "l2_scale": 0.01,
            "keep_prob": 0.5,
            "is_training": True,
            "early_stop_limit": 30
        }
    my_trainer = trainer.Trainer(my_data, my_model, epiml_options.logdir, **hparams)

    #train the model
    my_trainer.train()

    #outputs
    my_analyzer = analysis.Analysis(my_trainer)

    # my_analyzer.training_metrics()
    my_analyzer.validation_metrics()
    # my_analyzer.test_metrics()

    # outpath1 = os.path.join(epiml_options.logdir, "training_predict.csv")
    outpath2 = os.path.join(epiml_options.logdir, "validation_predict.csv")
    # outpath3 = os.path.join(epiml_options.logdir, "test_predict.csv")
    # my_analyzer.training_prediction(outpath1)
    my_analyzer.validation_prediction(outpath2)
    # my_analyzer.test_prediction(outpath3)

    my_analyzer.validation_confusion_matrix(epiml_options.logdir)
    # my_analyzer.test_confusion_matrix(epiml_options.logdir)

    # vis = visualization.Pca()
    # my_trainer.visualize(vis)

    # my_analyzer.importance() #TODO: generalize, probably put in model
    print('end {}'.format(datetime.datetime.now()))

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(sys.argv[1:])

