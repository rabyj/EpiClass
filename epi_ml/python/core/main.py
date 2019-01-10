import os
import sys
import warnings
warnings.simplefilter("ignore")
import data
import model
import trainer
import os.path
import numpy as np
import visualization
import datetime

import argparse
from argparseutils.directorytype import DirectoryType

def parse_arguments(args: list) -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('hdf5', type=argparse.FileType('r'), help='A file with hdf5 filenames. Use absolute path!')
    arg_parser.add_argument('chromsize', type=argparse.FileType('r'), help='A file with chrom sizes.')
    arg_parser.add_argument('metadata', type=argparse.FileType('r'), help='A metadata JSON file.')
    arg_parser.add_argument('logdir', type=DirectoryType(), help='A directory for the logs.')
    return arg_parser.parse_args(args)

def main(args):
    print('begin {}'.format(datetime.datetime.now()))
    epiml_options = parse_arguments(args)

    my_datasource = data.EpiDataSource(
        epiml_options.hdf5,
        epiml_options.chromsize,
        epiml_options.metadata)

    #test_path = "/Users/Jon/Projects/epi_ml/epi_ml/python/core/1110b4cbcd3f8659d2c479b8889e3eeb_1kb_all_none.hdf5"
    #test_path = "/Users/Jon/Projects/epi_ml/epi_ml/python/core/66638a9c6899bf55f60a8b95dca0eee4_1kb_all_none.hdf5"
    #spectro_hg(test_path)

    # my_data = data.EpiData("assay")
    my_data = data.EpiData(my_datasource , os.getenv('CATEGORY', 'assay'), oversample=True)
    #my_data = data.EpiData("publishing_group")

    #spectro(my_data.test.signals[55])

    input_size = my_data.train.signals[0].size
    ouput_size = my_data.train.labels[0].size

    my_model = model.Dense(input_size, ouput_size)
    #my_model = model.Cnn(41*49, ouput_size, (41, 49))
    #my_model = model.BidirectionalRnn(input_size, ouput_size)

    my_trainer = trainer.Trainer(my_data, my_model, epiml_options.logdir)
    my_trainer.train()
    my_trainer.metrics()
    print('end {}'.format(datetime.datetime.now()))

    # vis = visualization.Pca()
    # my_trainer.visualize(vis)
    #my_trainer.importance() #TODO: generalize, probably put in model

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(sys.argv[1:])
