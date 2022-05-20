"""Model creation module"""
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchinfo import summary
from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef, MetricCollection


class LightningDenseClassifier(pl.LightningModule): # pylint: disable=too-many-ancestors
    """Simple dense network handler"""

    def __init__(self, input_size, output_size, mapping, hparams, hl_units=3000, nb_layer=1):
        """Metrics expect probabilities and not logits"""
        super().__init__()
        # this is recommended Lightning way to save model arguments
        # it saves everything passed into __init__
        # and allows you to access it as self.myparam1, self.myparam2
        self.save_hyperparameters() #saves values given to __init__

        # -- general structure --
        self._x_size = input_size
        self._y_size = output_size
        self._hl_size = hl_units # hl = hidden layer
        self._nb_layer = nb_layer # number of intermediary/hidden layers

        self._mapping = mapping

        # -- hyperparameters --
        self.l2_scale = hparams.get("l2_scale", 0.01)
        self.dropout_rate = 1 - hparams.get("keep_prob", 0.5)
        self.learning_rate = hparams.get("learning_rate", 1e-5)

        self._pt_model = self.define_model()

        # Used Metrics
        metrics = MetricCollection([
            Accuracy(num_classes=self._y_size, average="micro"),
            Precision(num_classes=self._y_size, average="macro"),
            Recall(num_classes=self._y_size, average="macro"),
            F1Score(num_classes=self._y_size, average="macro"),
            MatthewsCorrCoef(num_classes=self._y_size)
            ])
        self.metrics = metrics
        self.train_acc = Accuracy(num_classes=self._y_size, average="micro")
        self.valid_acc = Accuracy(num_classes=self._y_size, average="micro")


    @property
    def mapping(self):
        """Return {output index:label} mapping."""
        return self._mapping

    @property
    def invert_mapping(self):
        """Return {label:output index} mapping."""
        return {val:key for key,val in self._mapping.items()}


    # --- Define general model structure ---
    def define_model(self):
        """ref : https://stackoverflow.com/questions/62937388/pytorch-dynamic-amount-of-layers"""
        layer_list = []

        # input layer
        layer_list.append(nn.Linear(self._x_size, self._hl_size))

        # hidden layers
        for _ in range(self._nb_layer - 1):
            layer_list.append(nn.Linear(self._hl_size, self._hl_size))
            # in case of ReLU, dropout should be applied before for computational efficiency
            # https://sebastianraschka.com/faq/docs/dropout-activation.html
            layer_list.append(nn.ReLU())
            layer_list.append(nn.Dropout(self.dropout_rate))

        # output layer
        layer_list.append(nn.Linear(self._hl_size, self._y_size))

        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
        model = nn.Sequential(*layer_list)

        return model

    def configure_optimizers(self):
        """https://pytorch.org/docs/stable/optim.html"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_scale
            )
        return optimizer


    # --- Define format of output ---
    def forward(self, x: torch.Tensor):
        """Return probabilities."""
        return F.softmax(self.forward_train(x), dim=1)

    def forward_train(self, x):
        """Return logits."""
        return self._pt_model(x)


    # --- Define how training and validation is done, what loss is used ---
    def training_step(self, train_batch, batch_idx):
        """Return training loss and co."""
        x, y = train_batch
        logits = self.forward_train(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        return {"loss": loss, "preds": preds.detach(), "target": y}

    def training_step_end(self, outputs):
        """Update and log training metrics."""
        self.train_acc(outputs["preds"], outputs["target"])

        # changing "step" for x-axis change
        metrics = {
            "train_acc": self.train_acc,
            "train_loss": outputs["loss"],
            "step": self.current_epoch + 1.0
            }
        self.log_dict(metrics, on_step=False, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        """Return validation loss and co."""
        x, y = val_batch
        logits = self.forward_train(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits, dim=1)
        return {"loss": loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):
        """Update and log validation metrics."""
        self.valid_acc(outputs["preds"], outputs["target"])

        # changing "step" for x-axis change
        metrics = {
            "valid_acc": self.valid_acc,
            "valid_loss": outputs["loss"],
            "step": self.current_epoch + 1.0
            }
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True)


    # --- Other information fonctions ---
    def print_model_summary(self, batch_size=1):
        """Print torchinfo summary."""
        print("--MODEL SUMMARY--")
        summary(
            model=self,
            input_size=(batch_size, self._x_size),
            col_names=["input_size", "output_size", "num_params"]
            )

    def compute_metrics(self, dataset):
        """Return dict of metrics for given dataset."""
        self.eval()
        with torch.no_grad():
            features, targets = dataset[:]
            preds = self(features)
        return self.metrics(preds, targets)

    def compute_predictions(self, dataset):
        """Return predictions and targets from dataset."""
        self.eval()
        with torch.no_grad():
            features, targets = dataset[:]
            preds = self(features)
        return preds, targets

    @classmethod
    def restore_model(cls, model_dir):
        """Load the checkpoint of the best model from the last run."""
        path = Path(model_dir) / "best_checkpoint.list"

        with open(path, "r", encoding="utf-8") as ckpt_file:
            lines = ckpt_file.read().splitlines()
            ckpt_path = lines[-1].split(' ')[0]

        print(f"Loading model from {ckpt_path}")
        return LightningDenseClassifier.load_from_checkpoint(ckpt_path)
