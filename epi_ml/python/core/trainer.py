import pytorch_lightning as pl
import pytorch_lightning.callbacks as torch_callbacks
import numpy as np
import pandas

import os.path
from scipy import signal
from abc import ABC
import math
from datetime import datetime

from .data import DataSet
from core.pytorch_model_test import LightningDenseClassifier

#TODO Print used hyperparameters
#TODO implement correct save/load (full save vs weight save only handling?)

class MyTrainer(pl.Trainer):

    def __init__(self, **kwargs):
        """Metrics expect probabilities and not logits"""
        super().__init__(**kwargs)

        self.best_checkpoint_file = os.path.join(
            self.log_dir,
            "best_checkpoint.list"
            )

    def save_model_path(self):
        """Save best checkpoint path to a file."""
        with open(self.best_checkpoint_file, "a") as ckpt_file:
            ckpt_file.write("{} {}\n".format(self.checkpoint_callback.best_model_path, datetime.now()))



def define_callbacks(early_stop_limit: int):
    """Returns list of PyTorch trainer callbacks.

    RichModelSummary, EarlyStopping, ModelCheckpoint, RichProgressBar
    """
    summary = torch_callbacks.RichModelSummary(max_depth=2)

    monitored_value="valid_loss"
    mode="min"

    early_stop = torch_callbacks.EarlyStopping(
        monitor=monitored_value,
        mode=mode,
        patience=early_stop_limit,
        check_on_train_epoch_end=False
    )

    checkpoint = torch_callbacks.ModelCheckpoint(
        monitor=monitored_value,
        mode=mode,
        save_last=True,
        auto_insert_metric_name=True,
        every_n_epochs=1,
        save_top_k=2,
        save_on_train_epoch_end=False
    )

    bar = torch_callbacks.RichProgressBar()

    return [summary, early_stop, checkpoint, bar]
