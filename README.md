# EpiClass - Epigenomic Classifier

Use machine learning to classify/label epigenomic data.

## Models availability - Neural networks trained on EpiATLAS

Models trained on the EpiATLAS dataset for inference on other datasets, which are associated with publication results, will be available on HuggingFace before said publication.

## Setup

The code was developed mainly on Python 3.8, and there is not guarantee for the requirements files to work on any other version.
A requirement file for Python 3.12 is also available, though some code paths were untested under that version.

The minimal requirements file is intended to provide what packages are needed to reproduce a training session. It does not
contain the required packages for analysis notebooks. For those, use `dev_requirements.txt` instead.

To install in a way that enables running the main training scripts:

- Clone the git
- Create a virtual environment, activate it
- In the code root directory (`epi_ml/src/python`, where `setup.py` is), run "`pip install -e .`"
- Install other requirements with pip `pip install -r ./requirements/minimal_requirements.txt`

The script used to automatically install the complete environment for training on HPC is `src/bash_utils/setup_venv.sh`.

If your system does not possess python3.8, you can use [pyenv](https://github.com/pyenv/pyenv) to manage multiple python versions.

## General

See `input-format` folder for examples of mandatory files.

See `src/bash_utils` folder for templates of launch scripts, with SLURM scheduler handling.

The main training scripts are in `src/python/epi_ml`.

`epiatlas_training.py` is meant for cross-validation training and evaluation, and `epiatlas_training_no_valid.py` is meant
for the training of a "complete" model without any cross-validation for a set amount of epochs. The template launcher `epiatlas_training.sh` can
handle both situations, depending on how the initial variables are modified.

`other_estimators.py` is used during the training and evaluation of non neural-network models.

Other scripts include:

- `predict.py` is for loading a trained model and making predictions on new data.

- `compute_shaps.py` is for computing SHAP values of a trained model on a specific set of data. It requires part of the training dataset
as input to properly compute SHAP values (this is the background). Ideally, the selected samples for SHAP background must be representative of the training dataset.

## Metadata handling and modifications

The `Metadata` class has an API to support modification of metadata as needed, like the `select_category_subsets` and `remove_category_subsets` methods. One can use them if they want to adjust label values on the fly, or perform additional filtering.

As soon as a label category exists in a dataset, any value is considered the label. Be it "", "--" or "NA".

If datasets containing differents keys is expected or possible, be sure to run `remove_missing_labels` on the relevant categories.

For additional information, refer to the [documentation](https://rabyj.github.io/epi_ml/epi_ml/python/core/metadata.html)

This API was constructed for the basic metadata handling needs of the training process.
For more control over the metadata, please use pandas instead.

## Command line interfaces of main scripts

For convenience, the command line interfaces of each script are provided here:

### epiatlas_training.py

~~~text
usage: epiatlas_training.py [-h] [--offline] [--restore] category hyperparameters hdf5 chromsize metadata logdir

positional arguments:
  category         The metatada category to analyse.
  hyperparameters  A json file containing model hyperparameters.
  hdf5             A file with hdf5 filenames. Use absolute path!
  chromsize        A file with chrom sizes.
  metadata         A metadata JSON file.
  logdir           Directory for the output logs.

options:
  -h, --help       show this help message and exit
  --offline        Will log data offline instead of online. Currently cannot merge comet-ml offline outputs.
  --restore        Skips training, tries to restore existing models in logdir for further analysis.
~~~

### epiatlas_training_no_valid.py

~~~text
usage: epiatlas_training_no_valid.py [-h] [--offline] [--restore]
                                     category hyperparameters hdf5 chromsize metadata logdir

positional arguments:
  category         The metatada category to analyse.
  hyperparameters  A json file containing model hyperparameters.
  hdf5             A file with hdf5 filenames. Use absolute path!
  chromsize        A file with chrom sizes.
  metadata         A metadata JSON file.
  logdir           Directory for the output logs.

options:
  -h, --help       show this help message and exit
  --offline        Will log data offline instead of online. Currently cannot merge comet-ml offline outputs.
  --restore        Skips training, tries to restore existing models in logdir for further analysis.
~~~

### predict.py

~~~text
usage: predict.py [-h] [--offline] [--model MODEL] hdf5 chromsize logdir

positional arguments:
  hdf5           A file with hdf5 filenames. Use absolute path!
  chromsize      A file with chrom sizes.
  logdir         Directory for the output logs.

options:
  -h, --help     show this help message and exit
  --offline      Will log data offline instead of online. Currently cannot merge comet-ml offline outputs.
  --model MODEL  Directory from which to load the desired model. Default is logdir.
~~~

### other_estimators.py

~~~text
usage: other_estimators.py [-h] [--models {all,LinearSVC,RF,LR,LGBM} [{all,LinearSVC,RF,LR,LGBM} ...]]
                           (--tune | --predict | --predict-new | --full-run) [-n N] [--hyperparams HYPERPARAMS]
                           category hdf5 chromsize metadata logdir

options:
  -h, --help            show this help message and exit

General:
  category              The metatada category to analyse.
  hdf5                  A file with hdf5 filenames. Use absolute path!
  chromsize             A file with chrom sizes.
  metadata              A metadata JSON file.
  logdir                Directory for the output logs.
  --models {all,LinearSVC,RF,LR,LGBM} [{all,LinearSVC,RF,LR,LGBM} ...]
                        Specify models to tune and/or predict.

Mode:
  --tune                Search best hyperparameters.
  --predict             FIT and PREDICT using hyperparameters.
  --predict-new         Use saved models to predict labels of new samples.
  --full-run            Tune then predict

Tune:
  -n N                  Number of BayesSearchCV hyperparameters iterations.

Predictions and Final training:
  --hyperparams HYPERPARAMS
                        A json file containing model(s) hyperparameters.
~~~

### compute_shaps.py

~~~text
usage: compute_shaps.py [-h] -m {NN,LGBM} --background_hdf5 background-hdf5 --explain_hdf5 explain-hdf5 --chromsize
                        CHROMSIZE [-l LOGDIR] [-o --output-name] [--model_file model_file] [--model_dir MODEL_DIR]

options:
  -h, --help            show this help message and exit

General arguments:
  -m {NN,LGBM}, --model {NN,LGBM}
                        Model to explain. Neural network or LightGBM.
  --background_hdf5 background-hdf5
                        A file with hdf5 filenames for the explainer background. Use absolute path!
  --explain_hdf5 explain-hdf5
                        A file with hdf5 filenames on which to compute SHAP values. Use absolute path!
  --chromsize CHROMSIZE
                        A file with chrom sizes.
  -l LOGDIR, --logdir LOGDIR
                        Directory for the output logs.
  -o --output-name, --output_name --output-name
                        Name (not path) of outputted pickle file containing computed SHAP value

Model dependant arguments:
  --model_file model_file
                        Needed for LGBM model. Specify the model file to load.
  --model_dir MODEL_DIR
                        Needed for neural network model. Directory with 'best_checkpoint.list' file.
~~~

## Tests

All tests are supposed to pass on tagged releases (on py3.8) when all requirements are installed, except

- two tests on standy (they take a long time)
- one unimplemented test

If you want to run tests, run pytest from `epi_ml/src/python/tests`.
