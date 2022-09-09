"""Script for debugging LightGBM."""
import json
import pickle
from pathlib import Path

import optuna
import optuna.integration.lightgbm as lgb
from lightgbm import log_evaluation, record_evaluation

optuna.logging.set_verbosity(optuna.logging.DEBUG)  # type: ignore

params = {
    "objective": "multiclass",
    "num_class": 4,
    "metric": ["multi_error", "multi_logloss", "auc_mu"],
    "boosting_type": "dart",
    "seed": 42,
    "num_threads": 4,
}

logdir = Path.cwd() / "debug"
with open(logdir / "input-n400-1split.pickle", "rb") as f:
    dsets = pickle.load(f)["dsets"]

dtrain = lgb.Dataset(  # type: ignore
    dsets.train.signals, label=dsets.train.encoded_labels, free_raw_data=True
)
dvalid = lgb.Dataset(  # type: ignore
    dsets.validation.signals,
    label=dsets.validation.encoded_labels,
    free_raw_data=True,
)

eval_results = {}
study = optuna.create_study(study_name="LGBM Classifier", direction="minimize")

tuner = lgb.LightGBMTuner(  # type: ignore
    params,
    dtrain,
    num_boost_round=5,
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


# import gzip

# with open(logdir / "input-n400.pickle.gz", "wb") as f:
#     data = pickle.dumps({"folds": folds, "dtrain": dtrain})
#     data = gzip.compress(data, 9)
#     f.write(data)

# exit()
