# EpiClass - Epigenomic Classifier

EpiClass trains machine learning models to classify and label epigenomic data.

## Model Availability – Neural Networks Trained on EpiATLAS

Models trained on the EpiATLAS dataset and used for inference on other datasets (as part of the associated publication) will be made available on HuggingFace prior to publication.

## Setup

The code was developed primarily with **Python 3.8**. Compatibility with other versions is not guaranteed. However, a separate requirements file is provided for **Python 3.12**. Note that some code paths remain untested under Python 3.12.

There are two requirements files:

- `requirements/minimal_requirements.txt`: Minimal packages needed to reproduce training.
- `dev_requirements.txt`: Additional packages for development, testing, and analysis notebooks.

To install the environment for training:

1. Clone this repository.
2. Create and activate a virtual environment.
3. From the Python code root (where `setup.py` is located), run:

```bash
pip install -e .
```

4. Then install the minimal requirements:

```bash
pip install -r requirements/minimal_requirements.txt
```

To automatically set up the training environment (e.g. on HPC clusters), use:

```bash
src/bash_utils/setup_venv.sh
```

If your system does not have Python 3.8 installed, consider using [pyenv](https://github.com/pyenv/pyenv) to manage multiple Python versions.

## Input Format & Job Launching

* See the `input-format/` folder for examples of required input files.
* The `src/bash_utils/` folder contains SLURM-compatible job launcher templates.
* Main training scripts are in `src/python/epi_ml/`.

### Key Scripts

* `epiatlas_training.py`: Performs cross-validation training and evaluation.
* `epiatlas_training_no_valid.py`: Trains the model without validation (e.g. final model for inference).
* `epiatlas_training.sh`: Job submission template supporting both training modes. Update variables as needed.
* `predict.py`: Uses a trained model to generate predictions on new data.
* `compute_shaps.py`: Computes SHAP values using a trained model and a representative background set.
* `other_estimators.py`: Trains and evaluates non-neural network models (e.g., Random Forest, LGBM, etc.).

## Metadata Handling

The `Metadata` class provides a convenient API for modifying metadata during preprocessing and training.

Notable methods:

* `select_category_subsets()`
* `remove_category_subsets()`

These allow dynamic relabeling or filtering of specific categories.

**Important notes:**

* Once a label category exists, any value (including `""`, `"--"`, or `"NA"`) is interpreted as a valid label.
* If your dataset may contain inconsistent keys, use `remove_missing_labels()` on the relevant categories.

For more details, refer to the [documentation](https://rabyj.github.io/epi_ml/epi_ml/python/core/metadata.html).

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

All tests are expected to pass on tagged releases (under Python 3.8), provided all requirements are installed.

Exceptions:

* Two tests are on standby (due to runtime).
* One test is currently unimplemented.

To run tests:

```bash
pytest src/python/tests
```

---

## License

This work is licensed under the GNU General Public License v3.0 (GPLv3)
