# EpiLaP - Epigenomic Label Predictor

Use machine learning on epigenomic data.

## Setup

To install in dev/editable mode:

- Clone the git
- Create a virtual environment, activate it
- In the root directory (setup.py), run "`pip install -e .`"
- Install other requirements with pip (see requirements folder).

## General

See `input-format` folder for examples of mandatory files.

~~~text
usage: main.py [-h] [--offline] [--predict] [--model MODEL]
               category hyperparameters hdf5 chromsize metadata logdir

positional arguments:
  category         The metatada category to analyse.
  hyperparameters  A json file containing model hyperparameters.
  hdf5             A file with hdf5 filenames. Use absolute path!
  chromsize        A file with chrom sizes.
  metadata         A metadata JSON file.
  logdir           Directory for the output logs.

optional arguments:
  -h, --help       show this help message and exit
  --offline        Will log data offline instead of online. Currently cannot merge comet-ml offline
                   outputs.
  --predict        Enter prediction mode. Will use all data for the test set. Overwrites hparameter file
                   setting. Default mode is training mode.
  --model MODEL    Directory from which to load the desired model. Default is logdir.
~~~

## Metadata handling and modifications

The `Metadata` class has an api to support modification of metadata as needed, like the `select_category_subsets` and `remove_category_subsets` methods. One can use them if they want to adjust label values on the fly, or perform additional filtering.

As soon as a label category exists in a dataset, any value is considered the label. Be it "", "--" or "NA".

If datasets containing differents keys is expected or possible, be sure to run `remove_missing_labels` on the relevant categories.
