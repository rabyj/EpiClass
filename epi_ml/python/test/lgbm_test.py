import gzip
import pathlib
import pickle

import optuna.integration.lightgbm as lgb

params = {
    "objective": "multiclass",
    "num_class": 4,
    "metric": "multi_logloss",
    "boosting_type": "dart",
    "seed": 42,
    "verbosity": 4,
    "num_threads": 4,
}

with gzip.open(
    pathlib.Path(__file__).parent.resolve() / "input-n40.pickle.gz", "rb"
) as f:
    input_data = pickle.load(f)

dtrain = input_data["dtrain"]
folds = input_data["folds"]

eval_results = {}

tuner = lgb.LightGBMTunerCV(
    params,
    dtrain,
    folds=folds,
    callbacks=[lgb.record_evaluation(eval_results)],
    show_progress_bar=False,
)


tuner.run()
