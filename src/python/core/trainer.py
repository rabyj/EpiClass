"""Trainer class extensions module"""
from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
import pytorch_lightning.callbacks as torch_callbacks


class MyTrainer(pl.Trainer):
    """Personalized trainer"""

    def __init__(self, general_log_dir: str, last_trained_model=None, **kwargs):
        """Metrics expect probabilities and not logits."""
        super().__init__(**kwargs)

        self.best_checkpoint_file = Path(general_log_dir) / "best_checkpoint.list"
        self.model = last_trained_model
        self.batch_size = None

    def fit(self, *args, verbose=True, **kwargs):
        """Base pl.Trainer.fit function, but also prints the batch size."""
        self.batch_size = kwargs["train_dataloaders"].batch_size
        if verbose:
            print(f"Training batch size : {self.batch_size}")
        super().fit(*args, **kwargs)

    def save_model_path(self):
        """Save best checkpoint path to a file."""
        print(f"Saving model to {self.checkpoint_callback.best_model_path}")
        with open(self.best_checkpoint_file, "a", encoding="utf-8") as ckpt_file:
            ckpt_file.write(
                f"{self.checkpoint_callback.best_model_path} {datetime.now()}\n"
            )

    def print_hyperparameters(self):
        """Print training hyperparameters."""
        stop_callback = self.early_stopping_callback
        print("--TRAINING HYPERPARAMETERS--")
        print(f"L2 scale : {self.model.l2_scale}")
        print(f"Dropout rate : {self.model.dropout_rate}")
        print(f"Learning rate : {self.model.learning_rate}")
        print(f"Patience : {stop_callback.patience}")
        print(f"Monitored value : {stop_callback.monitor}")


def define_callbacks(early_stop_limit: int, show_summary=True):
    """Returns list of PyTorch trainer callbacks.
    RichModelSummary, EarlyStopping, ModelCheckpoint
    """
    callbacks = []
    if show_summary:
        callbacks.append(torch_callbacks.RichModelSummary(max_depth=3))

    monitored_value = "valid_acc"  # have same name as TorchMetrics
    mode = "max"

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

    return callbacks
