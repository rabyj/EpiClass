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

    from epiclass.argparseutils.DefaultHelpParser import (
        DefaultHelpParser as ArgumentParser,
    )
    from epiclass.argparseutils.directorychecker import DirectoryChecker
    from epiclass.core import analysis, data, estimators, metadata
    from epiclass.core.data import DataSet, UnknownData
    from epiclass.core.data_source import EpiDataSource
    from epiclass.core.epiatlas_treatment import EpiAtlasFoldFactory
    from epiclass.core.hdf5_loader import Hdf5Loader
    from epiclass.core.lgbm import tune_lgbm
    from epiclass.core.model_pytorch import LightningDenseClassifier
    from epiclass.core.trainer import MyTrainer, define_callbacks
    from epiclass.utils.check_dir import create_dirs
    from epiclass.utils.time import time_now
