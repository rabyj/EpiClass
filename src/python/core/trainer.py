"""Trainer class extensions module"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.callbacks as torch_callbacks


class MyTrainer(pl.Trainer):
    """Personalized trainer"""

    def __init__(self, general_log_dir: str, model, **kwargs):
        """Metrics expect probabilities and not logits."""
        super().__init__(**kwargs)

        self.best_checkpoint_file = Path(general_log_dir) / "best_checkpoint.list"
        self.my_model = model
        self.batch_size = None

    def fit(self, *args, verbose=True, **kwargs):
        """Base pl.Trainer.fit function, but also prints the batch size."""
        self.batch_size = kwargs["train_dataloaders"].batch_size
        if verbose:
            print(f"Training batch size : {self.batch_size}")
        super().fit(*args, **kwargs)

    def save_model_path(self):
        """Save best checkpoint path to a file."""
        try:
            model_path = self.checkpoint_callback.best_model_path  # type: ignore
            print(f"Saving model to {model_path}")
            with open(self.best_checkpoint_file, "a", encoding="utf-8") as ckpt_file:
                ckpt_file.write(f"{model_path} {datetime.now()}\n")
        except AttributeError:
            print("Cannot save model, no checkpoint callback.")

    def print_hyperparameters(self):
        """Print training hyperparameters."""
        print("--TRAINING HYPERPARAMETERS--")
        print(f"L2 scale : {self.my_model.l2_scale}")
        print(f"Dropout rate : {self.my_model.dropout_rate}")
        print(f"Learning rate : {self.my_model.learning_rate}")
        try:
            stop_callback = self.early_stopping_callback
            print(f"Patience : {stop_callback.patience}")  # type: ignore
            print(f"Monitored value : {stop_callback.monitor}")  # type: ignore
        except AttributeError:
            print("No early stopping.")


def define_callbacks(early_stop_limit: int | None, show_summary=True):
    """Returns list of PyTorch trainer callbacks.
    RichModelSummary, EarlyStopping, ModelCheckpoint

    Will only save last epoch model if there is no early stopping.
    """
    callbacks = []
    if show_summary:
        callbacks.append(torch_callbacks.RichModelSummary(max_depth=3))

    monitored_value = "valid_acc"  # have same name as TorchMetrics
    mode = "max"

    if early_stop_limit is not None:
        callbacks.append(
            torch_callbacks.EarlyStopping(
                monitor=monitored_value,
                mode=mode,
                patience=early_stop_limit,
                check_on_train_epoch_end=False,
            )
        )

        callbacks.append(
            torch_callbacks.ModelCheckpoint(
                monitor=monitored_value,
                mode=mode,
                save_last=True,
                auto_insert_metric_name=True,
                every_n_epochs=1,
                save_top_k=2,
                save_on_train_epoch_end=False,
            )
        )
    else:
        callbacks.append(
            torch_callbacks.ModelCheckpoint(
                monitor=None,
            )
        )

    return callbacks
