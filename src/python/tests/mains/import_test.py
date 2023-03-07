"""Test if all main scripts imports run without errors."""
# pylint: disable=unused-import, import-outside-toplevel
from __future__ import annotations


def import_test():
    """Test if all main scripts imports run without errors."""
    import argparse
    import glob
    import json
    import os
    import sys
    import warnings
    from functools import partial
    from pathlib import Path
    from typing import Dict

    import comet_ml  # needed because special snowflake # pylint: disable=unused-import
    import numpy as np
    import pytorch_lightning as pl  # in case GCC or CUDA needs it # pylint: disable=unused-import
    import pytorch_lightning.callbacks as pl_callbacks
    import torch
    from pytorch_lightning import loggers as pl_loggers
    from torch.utils.data import DataLoader, TensorDataset

    from epi_ml.argparseutils.DefaultHelpParser import DefaultHelpParser as ArgumentParser
    from epi_ml.argparseutils.directorychecker import DirectoryChecker
    from epi_ml.core import analysis, data, estimators, metadata
    from epi_ml.core.data import DataSet, UnknownData
    from epi_ml.core.data_source import EpiDataSource
    from epi_ml.core.epiatlas_treatment import EpiAtlasFoldFactory
    from epi_ml.core.hdf5_loader import Hdf5Loader
    from epi_ml.core.lgbm import tune_lgbm
    from epi_ml.core.model_pytorch import LightningDenseClassifier
    from epi_ml.core.trainer import MyTrainer, define_callbacks
    from epi_ml.utils.check_dir import create_dirs
    from epi_ml.utils.time import time_now
