import os
import sys
import warnings
warnings.simplefilter("ignore")
import os.path
import numpy as np
import datetime

import data
import model
import trainer
import figs
import visualization

import argparse
from argparseutils.directorytype import DirectoryType

def parse_arguments(args: list) -> argparse.Namespace:
    """argument parser for command line"""
    arg_parser = argparse.ArgumentParser()
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
    # figs.convert_matrix_csv_to_png(in_path, out_path)
    # sys.exit()

    #load metadata
    my_datasource = data.EpiDataSource(
        epiml_options.hdf5,
        epiml_options.chromsize,
        epiml_options.metadata)

    #load data
    my_data = data.EpiData(my_datasource , os.getenv('CATEGORY', 'assay'), oversample=False, min_class_size=3)
    my_data.display_labels()

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
            "training_epochs": 50,
            "batch_size": 256,
            "measure_frequency": 1,
            "l1_scale": 0.001, #ONLY IN L1DENSE
            "l2_scale": 0.01,
            "keep_prob": 0.5,
            "is_training": True,
            "early_stop_limit": 5
        }
    my_trainer = trainer.Trainer(my_data, my_model, epiml_options.logdir, **hparams)
    #train the model
    my_trainer.train()
    #outputs
    my_trainer.metrics()
    my_trainer.confusion_matrix()
    # vis = visualization.Pca()
    # my_trainer.visualize(vis)
    #my_trainer.importance() #TODO: generalize, probably put in model
    print('end {}'.format(datetime.datetime.now()))

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(sys.argv[1:])
