# EpiClass - Epigenomic Classifier

EpiClass trains machine learning models to classify and label epigenomic data.

## Publication

This repository contains most of the code used to obtain results for the following paper:
[Leveraging the largest harmonized epigenomic data collection for metadata prediction validated and augmented over 350,000 public epigenomic datasets](https://doi.org/10.1101/2025.09.04.670545)

To interact with the paper figures, use the [Quarto website](https://rabyj.github.io/EpiClass/epiclass-paper/index.html). This website is generated from an alternative version of the Python notebooks used to create the figures (notebooks are at `src/python/epiclass/utils/notebooks/paper/paper-final/fig*.ipynb`).

See [Key Scripts](#key-scripts) section for the training code.

## Model Availability – Neural Networks Trained on EpiATLAS

Models trained on the EpiATLAS dataset and used for inference on other datasets (as part of the associated publication) are available on HuggingFace under the ["EpiClass models" collection](https://huggingface.co/collections/KatLeChat/epiclass-models-68adb5ce65c8f2fb93322e59).

## Setup

The code was developed primarily with **Python 3.8**. Compatibility with other versions is not guaranteed. However, the test suite passed under python 3.9 to 3.11.

To install the environment for training:

1. Clone this repository.
2. Create and activate a virtual environment.
3. From the Python code root (where `pyproject.toml` is located), run:

```bash
pip install -e . # you can also use 'uv'
```

To install the environment for analysis notebooks or running tests:

```bash
pip install -e .[utils] # adds requirements notebooks and utility scripts
pip install -e .[test] # adds pytest requirements
```

The exact requirements are in `src/python/requirements/req_core.in`. Other dependencies are specified in `pyproject.toml`.

## Input Format & Job Launching

- See the `input-format/` folder for examples of required input files.
- The `src/bash_utils/` folder contains SLURM-compatible job launcher templates.
- Main training scripts are in `src/python/epiclass/`.

### Key Scripts

- `epiatlas_training.py`: Performs cross-validation training and evaluation.
- `epiatlas_training_no_valid.py`: Trains the model without validation (e.g. final model for inference).
- `epiatlas_training.sh`: Job submission template supporting both training modes. Update variables as needed.
- `predict.py`: Uses a trained model to generate predictions on new data.
- `compute_shaps.py`: Computes SHAP values using a trained model and a representative background set.
- `other_estimators.py`: Trains and evaluates non-neural network models (e.g., Random Forest, LGBM, etc.).

## Metadata Handling

The `Metadata` class provides a convenient API for modifying metadata during preprocessing and training.

Notable methods:

- `select_category_subsets()`
- `remove_category_subsets()`

These allow dynamic relabeling or filtering of specific categories.

**Important notes:**

- Once a label category exists, any value (including `""`, `"--"`, or `"NA"`) is interpreted as a valid label.
- If your dataset may contain inconsistent keys, use `remove_missing_labels()` on the relevant categories.

For more details, refer to the [documentation](https://rabyj.github.io/EpiClass/epiclass/core/metadata.html).

For advanced metadata manipulation, use `pandas` directly.

## Command-Line Interfaces

### `epiatlas_training.py`

```text
usage: epiatlas_training.py [-h] [--offline] [--restore]
                            category hyperparameters hdf5 chromsize metadata logdir

positional arguments:
  category         The metadata category to analyze.
  hyperparameters  JSON file containing model hyperparameters.
  hdf5             File with HDF5 paths (use absolute paths).
  chromsize        Chromosome sizes file.
  metadata         Metadata JSON file.
  logdir           Output directory.

options:
  -h, --help       Show this help message and exit.
  --offline        Use offline logging. (Note: Comet-ML offline logs cannot currently be merged.)
  --restore        Skip training; restore and reuse existing models from logdir.
```

### `epiatlas_training_no_valid.py`

```text
usage: epiatlas_training_no_valid.py [-h] [--offline] [--restore]
                                     category hyperparameters hdf5 chromsize metadata logdir

(Same arguments and options as above.)
```

### `predict.py`

The model directory should be the folder where the checkpoint `best_checkpoint.list` list is.
The last path of this file will be loaded, so make sure the path points to a model weights file (`.ckpt`) that exists.

```text
usage: predict.py [-h] [--offline] [--model MODEL] hdf5 chromsize logdir

positional arguments:
  hdf5           File with HDF5 paths.
  chromsize      Chromosome sizes file.
  logdir         Output directory.

options:
  -h, --help     Show this help message and exit.
  --offline      Use offline logging.
  --model MODEL  Directory containing the model (defaults to `logdir`).
```

### `other_estimators.py`

```text
usage: other_estimators.py [-h]
       [--models {all,LinearSVC,RF,LR,LGBM} [{all,LinearSVC,RF,LR,LGBM} ...]]
       (--tune | --predict | --predict-new | --full-run)
       [-n N] [--hyperparams HYPERPARAMS]
       category hdf5 chromsize metadata logdir

positional arguments:
  category              Metadata category to analyze.
  hdf5                  File with HDF5 paths.
  chromsize             Chromosome sizes file.
  metadata              Metadata JSON file.
  logdir                Output directory.

options:
  -h, --help            Show this help message and exit.
  --models              Specify models to use. Choices: all, LinearSVC, RF, LR, LGBM.

Modes:
  --tune                Perform hyperparameter search.
  --predict             Fit model(s) and predict.
  --predict-new         Use saved models to predict new samples.
  --full-run            Tune and then predict.

Tuning:
  -n N                  Number of iterations for BayesSearchCV.

Prediction:
  --hyperparams         JSON file with model hyperparameters.
```

### `compute_shaps.py`

```text
usage: compute_shaps.py [-h]
       -m {NN,LGBM}
       --background_hdf5 BACKGROUND_HDF5
       --explain_hdf5 EXPLAIN_HDF5
       --chromsize CHROMSIZE
       [-l LOGDIR]
       [-o OUTPUT_NAME]
       [--model_file MODEL_FILE]
       [--model_dir MODEL_DIR]

Required arguments:
  -m, --model           Model type: NN (neural network) or LGBM.
  --background_hdf5     File with background HDF5s (for SHAP explainer). Absolute path required.
  --explain_hdf5        File with HDF5s to explain. Absolute path required.
  --chromsize           Chromosome sizes file.

Optional arguments:
  -l, --logdir          Output directory.
  -o, --output_name     Output filename (pickle) for SHAP values.

Model-specific:
  --model_file          Path to trained LGBM model.
  --model_dir           Directory with NN model (`best_checkpoint.list` required).
```

## Tests

All tests are expected to pass on tagged releases since v0.3.0, provided all requirements are installed.

Exceptions:

- Three tests are on standby (due to runtime).
- One test is currently unimplemented.

To run tests, first uncompress fixtures `src/python/tests/fixtures.tar.xz` as folder `fixtures`.

```bash
pytest src/python/tests
```

---

## License

This work is licensed under the GNU General Public License v3.0 (GPLv3)
