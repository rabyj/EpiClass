"""Module to define how LightGBM is handled."""
import json
import pickle
from pathlib import Path

import optuna
import optuna.integration.lightgbm as lgb
import pandas as pd
from lightgbm import log_evaluation

from epi_ml.python.core.epiatlas_treatment import EpiAtlasTreatment


def tune_lgbm(ea_handler: EpiAtlasTreatment, logdir: Path):
    """
    It takes an EpiAtlasTreatment object and a log directory, and it tunes the
    hyperparameters of a LightGBM classifier using Optuna.

    Args:
      ea_handler (EpiAtlasTreatment): Dataset
      logdir (Path): The directory where the results will be saved.

    Returns:
      The Optuna study object.
    """
    params = {
        "objective": "multiclass",
        "num_class": len(ea_handler.classes),
        "metric": ["multi_logloss", "multi_error"],
        "boosting_type": "dart",
        "seed": 42,
        "force_col_wise": True,
        "device_type": "cpu",
        "num_threads": 9,
    }

    # Using first fold to tune hyperparams
    dsets = next(ea_handler.yield_split())

    dtrain = lgb.Dataset(  # type: ignore
        dsets.train.signals, label=dsets.train.encoded_labels, free_raw_data=True
    )
    dvalid = lgb.Dataset(  # type: ignore
        dsets.validation.signals,
        label=dsets.validation.encoded_labels,
        free_raw_data=True,
    )
    dsets = None
    del dsets

    # Create tuner
    max_iter = 100  # boosting rounds per trial
    study = optuna.create_study(
        study_name="LGBM Classifier",
    )

    tuner = lgb.LightGBMTuner(  # type: ignore
        params,
        dtrain,
        num_boost_round=max_iter,
        study=study,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            log_evaluation(period=5),  # type: ignore
        ],
        show_progress_bar=False,
    )

    # Optimize and log results.
    print("Tuning feature fraction")
    tuner.tune_feature_fraction(3)
    print("Tuning number of leaves")
    tuner.tune_num_leaves(5)
    print("Tuning bagging")
    tuner.tune_bagging(4)

    print("Best score:", tuner.best_score)

    best_params = tuner.best_params
    print("Best params:", best_params)
    print("  Params: ")
    for key, value in list(best_params.items()):
        print(f"    {key}: {value}")

        # Put it in the pipeline format
        del best_params[key]
        best_params[f"model__{key}"] = value

    name = "LGBM"
    with open(logdir / f"{name}_best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f)

    with open(logdir / f"{name}_study.pickle", "wb") as f:
        pickle.dump(study, f)

    df = pd.DataFrame(study.trials_dataframe())
    df.to_csv(logdir / f"{name}_trials.csv")

    return study
