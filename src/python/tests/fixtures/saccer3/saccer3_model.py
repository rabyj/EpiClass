"""Train and save a saccer3 assay model for testings purposes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from epi_ml.main import main as epi_ml_main


def main():
    """Create saccer3 assay model."""
    current_dir = Path(__file__).parent.resolve()
    logdir = current_dir

    category = "assay"
    hparams = "saccer3_hparams.json"
    hdf5_list = "hdf5_10kb_all_none.list"
    chrom_file = "saccer3.can.chrom.sizes"
    metadata_file = "saccer3_2016-07_metadata.json"

    os.environ["MIN_CLASS_SIZE"] = "3"
    os.environ["LAYER_SIZE"] = "1500"
    os.environ["NB_LAYER"] = "1"

    sys.argv = [
        "main.py",
        category,
        str(logdir / hparams),
        str(logdir / hdf5_list),
        str(logdir / chrom_file),
        str(logdir / metadata_file),
        str(logdir),
    ]

    epi_ml_main()


if __name__ == "__main__":
    main()
