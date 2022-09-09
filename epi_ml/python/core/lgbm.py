"""Module to define how LightGBM is handled."""
import json
from pathlib import Path

import optuna
import optuna.integration.lightgbm as lgb
from lightgbm import log_evaluation, record_evaluation

from epi_ml.python.core.epiatlas_treatment import EpiAtlasTreatment


def tune_lgbm(ea_handler: EpiAtlasTreatment, logdir: Path):
    """
    It takes an EpiAtlasTreatment object and a log directory, and it tunes the
    hyperparameters of a LightGBM classifier using Optuna.

    Args:
      ea_handler (EpiAtlasTreatment): Dataset
      logdir (Path): The directory where the results will be saved.

    Returns:
      The evaluation history.
    """
    params = {
        "objective": "multiclass",
        "num_class": len(ea_handler.classes),
        "metric": ["multi_error", "multi_logloss", "auc_mu"],
        "boosting_type": "dart",
        "seed": 42,
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

    # Create tuner
    max_iter = 100  # boosting rounds per trial
    eval_results = {}
    study = optuna.create_study(
        study_name="LGBM Classifier",
        direction="minimize",
    )

    tuner = lgb.LightGBMTuner(  # type: ignore
        params,
        dtrain,
        num_boost_round=max_iter,
        study=study,
        valid_sets=[dvalid],
        valid_names=["valid"],
        callbacks=[
            record_evaluation(eval_results),  # type: ignore
            log_evaluation(period=1),  # type: ignore
        ],
        show_progress_bar=True,
    )

    # Optimize and log results.
    tuner.run()

    print("Best score:", tuner.best_score)
    best_params = tuner.best_params
    print("Best params:", best_params)
    print("  Params: ")
    for key, value in best_params.items():
        print(f"    {key}: {value}")

    name = "LGBM"
    with open(logdir / f"{name}_optim.json", "w", encoding="utf-8") as f:
        json.dump(obj=eval_results, fp=f)

    with open(logdir / f"{name}_best_params.json", "w", encoding="utf-8") as f:
        json.dump(obj=best_params, fp=f)

    return eval_results
