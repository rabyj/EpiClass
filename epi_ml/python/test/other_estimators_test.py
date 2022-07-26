"""Main"""
import json
import os
import sys

from epi_ml.python.core import metadata
from epi_ml.python.core.data_source import EpiDataSource
from epi_ml.python.test.core.epiatlas_treatment_test import EpiAtlasTreatment as EpiTest
from epi_ml.python.core.epiatlas_treatment import EpiAtlasTreatment as EpiAtlasTreatment

to_exclude = ["main"]
from epi_ml.python.other_estimators import * # pylint: disable=wildcard-import
for name in to_exclude:
    del globals()[name]


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
    i = 0
    with open("test.md5s", 'w', encoding="utf-8") as f:
        for md5s in test.values():

            for md5 in md5s:
                f.write(f"{md5}\n")

                #other matching tracks
                for rest in md5_mapping[md5].values():
                    f.write(f"{rest}\n")

                i += 1
                if i % 50 == 0:
                    break



def main(args): # pylint: disable=function-redefined
    """main called from command line, edit to change behavior"""
    begin = time_now()
    print(f"begin {begin}")

    # --- PARSE params and LOAD external files ---
    cli = parse_arguments(args)

    my_datasource = EpiDataSource(
        cli.hdf5,
        cli.chromsize,
        cli.metadata
        )

    my_metadata = metadata.Metadata(my_datasource.metadata_file)
    my_metadata.remove_category_subsets(label_category="track_type", labels=["Unique.raw"])

    if os.getenv("EXCLUDE_LIST") is not None:
        exclude_list = json.loads(os.environ["EXCLUDE_LIST"])
        my_metadata.remove_category_subsets(label_category=cli.category, labels=exclude_list)

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
    # exit()


    # -- debugging time --
    loading_begin = time_now()

    # with open(cli.hdf5.parent / "estimator-test.md5s", 'r', encoding="utf-8") as md5_file:
    #     chosen_md5s = {line.rstrip() for line in md5_file}

    # for md5 in list(my_metadata.md5s):
    #     if md5 not in chosen_md5s:
    #         del my_metadata[md5]

    ea_handler = EpiTest(my_datasource, cli.category, assay_list,
    n_fold=2, test_ratio=0, min_class_size=min_class_size, meta=None
    )

    loading_time = time_now() - loading_begin
    print(f"Initial hdf5 loading time: {loading_time}")

    n_iter = cli.n
    if cli.only_svm:
        optimize_svm(ea_handler, cli.logdir, n_iter)  # type: ignore
        sys.exit()
    elif cli.only_rf:
        optimize_rf(ea_handler, cli.logdir, n_iter) # type: ignore
        sys.exit()
    else:
        optimize_rf(ea_handler, cli.logdir, n_iter) # type: ignore
        optimize_svm(ea_handler, cli.logdir, n_iter) # type: ignore


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    main(sys.argv[1:])
