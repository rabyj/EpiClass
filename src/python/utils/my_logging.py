"""Implement logging of specific data through comet logger."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict

from pytorch_lightning import loggers as pl_loggers


def log_pre_training(
    logger: pl_loggers.CometLogger, to_log: Dict[str, str], step: int | None
):
    """Log a bunch of stuff in comet logger. Return experience key (str).

    to_log expects keys:
    - category
    - hdf5_resolution
    - loading_time (initial, for hdf5)
    - SLURM_JOB_ID (optional)
    When step is an int:
    - split_time (generator yield time)
    """
    # General stuff
    category = to_log.get("category")
    logger.experiment.add_tag(category)
    logger.experiment.log_other("category", category)

    logger.experiment.log_other(
        "HDF5 Resolution", f"{int(to_log['hdf5_resolution'])/1000}kb"
    )

    # job id when done on HPC
    if "SLURM_JOB_ID" in to_log:
        logger.experiment.log_other("SLURM_JOB_ID", os.environ["SLURM_JOB_ID"])
        logger.experiment.add_tag("Cluster")

    # Code time stuff
    loading_time = to_log.get("loading_time")
    print(f"Initial hdf5 loading time: {loading_time}")
    logger.experiment.log_other("Initial hdf5 loading time", loading_time)

    if step is not None:
        split_time = to_log.get("split_time")
        print(f"Set loading/splitting time: {split_time}")
        logger.experiment.log_metric("Split_time", split_time, step=step)

    # exp id
    exp_key = logger.experiment.get_key()
    print(f"The current experiment key is {exp_key}")
    logger.experiment.log_other("Experience key", str(exp_key))

    # Save git commit.
    os.chdir(Path(__file__).resolve().parent)
    commit_label = subprocess.check_output(["git", "describe"], encoding="UTF-8").strip()
    print(f"The current commit is {commit_label}")
    logger.experiment.log_other("Code version / commit", f"{commit_label}")
