"""Main"""
import json
import os
from pathlib import Path

import optuna
import pandas as pd
from tabulate import tabulate

import src.python.core.estimators as estimators
import src.python.other_estimators as estimators_main
from src.python.core import lgbm, metadata
from src.python.core.data_source import EpiDataSource
from src.python.core.epiatlas_treatment import EpiAtlasTreatment as EpiAtlasTreatment
from src.python.test.core.epiatlas_treatment_test import EpiAtlasTreatment as EpiTest
from src.python.utils.time import time_now


def optimize_svm(ea_handler, logdir: Path):
    """Optimize an sklearn SVC over a hyperparameter space."""
    print("Starting SVM optimization")
    start_train = time_now()
    opt = estimators.tune_estimator(
        estimators.model_mapping["LinearSVC"],
        ea_handler,
        estimators.SVM_LIN_SEARCH,
        n_iter=1,
        concurrent_cv=1,
        n_jobs=1,
    )
    print(f"Total linear SVM optimisation time: {time_now()-start_train}")

    df = pd.DataFrame(opt.cv_results_)
    print(tabulate(df, headers="keys", tablefmt="psql"))  # type: ignore
    df.to_csv(logdir / "SVM_lin_optim.csv", sep=",")


def create_test_list(ea_handler: EpiTest):
    """Create a small list of files with 50 'leading' signals and their match for each
    biomaterial type.
    """
    meta = ea_handler.get_complete_metadata(verbose=True)
    md5_mapping = ea_handler.group_mapper
    md5_per_class = meta.md5_per_class("biomaterial_type")

    # Seek "leading" tracks
    test = {}
    for label_class in meta.unique_classes("biomaterial_type"):
        test[label_class] = []
        for md5 in md5_per_class[label_class]:
            if md5 in md5_mapping:
                test[label_class].append(md5)

    # Write md5s of groups to file
    md5_list = set({})
    for md5s in test.values():
        i = 0
        for md5 in md5s:
            md5_list.add(md5)

            # other matching tracks
            for rest in md5_mapping[md5].values():
                md5_list.add(rest)

            i += 1
            if i % 100 == 0:
                break

    with open("test.md5s", "w", encoding="utf-8") as f:
        for md5 in md5_list:
            f.write(f"{md5}\n")

    return md5_list


def main():  # pylint: disable=function-redefined
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = estimators_main.parse_arguments()

    my_datasource = EpiDataSource(cli.hdf5, cli.chromsize, cli.metadata)

    my_metadata = metadata.Metadata(my_datasource.metadata_file)
    my_metadata.remove_category_subsets(
        label_category="track_type", labels=["Unique.raw"]
    )

    if os.getenv("EXCLUDE_LIST") is not None:
        exclude_list = json.loads(os.environ["EXCLUDE_LIST"])
        my_metadata.remove_category_subsets(
            label_category=cli.category, labels=exclude_list
        )

    if os.getenv("ASSAY_LIST") is not None:
        assay_list = json.loads(os.environ["ASSAY_LIST"])
        print(f"Going to only keep targets with {assay_list}")
    else:
        assay_list = my_metadata.unique_classes(cli.category)
        print("No assay list")

    if os.getenv("MIN_CLASS_SIZE") is not None:
        min_class_size = int(os.environ["MIN_CLASS_SIZE"])
    else:
        min_class_size = 10

    #  -- test list creation --
    # my_datasource = EpiDataSource(
    #     cli.hdf5.parent / "100kb_all_none_plus.list",
    #     cli.chromsize,
    #     cli.metadata
    #     )

    # ea_handler = EpiTest(my_datasource, cli.category, assay_list,
    # n_fold=NFOLD, test_ratio=0, min_class_size=min_class_size
    # )
    # create_test_list(ea_handler) # type: ignore

    # -- debugging time --
    loading_begin = time_now()

    ea_handler = EpiAtlasTreatment(
        my_datasource,
        cli.category,
        assay_list,
        n_fold=2,
        test_ratio=0,
        min_class_size=min_class_size,
    )

    loading_time = time_now() - loading_begin
    print(f"Initial hdf5 loading time: {loading_time}")

    n_iter = cli.n
    if "all" in cli.models:
        models = estimators.model_mapping.keys()
    else:
        models = cli.models

    for name in models:
        if name == "LGBM":
            optuna.logging.set_verbosity(optuna.logging.DEBUG)  # type: ignore
            lgbm.tune_lgbm(ea_handler, cli.logdir)
        else:
            estimators.optimize_estimator(ea_handler, cli.logdir, n_iter, name)


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main()
