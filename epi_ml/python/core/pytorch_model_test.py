import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchinfo import summary
from torchmetrics import Accuracy, Precision, Recall, F1Score, MatthewsCorrCoef, MetricCollection

class LightningDenseClassifier(pl.LightningModule):

    def __init__(self, input_size, output_size, hparams, hl_units=3000, nb_layer=1):
        """Metrics expect probabilities and not logits"""
        super(LightningDenseClassifier, self).__init__()

        # -- general structure --
        self._x_size = input_size
        self._y_size = output_size
        self._hl_size = hl_units # hl = hidden layer
        self._nb_layer = nb_layer # number of intermediary/hidden layers

        # -- hyperparameters --
        self._l2_scale = hparams.get("l2_scale", 0.01)
        self._dropout_rate = 1 - hparams.get("keep_prob", 0.5)
        self._learning_rate = hparams.get("learning_rate", 1e-5)

        self._pt_model = self.define_model()

        # Used Metrics.
        # metrics = MetricCollection([
        #     Accuracy(num_classes=self._y_size, average="micro"),
        #     Precision(num_classes=self._y_size, average="macro"),
        #     Recall(num_classes=self._y_size, average="macro"),
        #     F1Score(num_classes=self._y_size, average="macro"),
        #     MatthewsCorrCoef(num_classes=self._y_size)
        #     ])
        # self.train_metrics = metrics.clone(prefix='train_')
        # self.valid_metrics = metrics.clone(prefix='valid_')
        # self.train_acc = Accuracy(num_classes=self._y_size, average="micro")
        # self.valid_acc = Accuracy(num_classes=self._y_size, average="micro")

    def define_model(self):
        """ref : https://stackoverflow.com/questions/62937388/pytorch-dynamic-amount-of-layers"""
        layer_list = []

        # input layer
        layer_list.append(torch.nn.Linear(self._x_size, self._hl_size))

        # hidden layers
        for _ in range(self._nb_layer - 1):
            layer_list.append(torch.nn.Linear(self._hl_size, self._hl_size))
            # in case of ReLU, dropout should be applied before for computational efficiency
            # https://sebastianraschka.com/faq/docs/dropout-activation.html
            layer_list.append(torch.nn.ReLU())
            layer_list.append(torch.nn.Dropout(self._dropout_rate))

        # output layer
        layer_list.append(torch.nn.Linear(self._hl_size, self._y_size))

        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
        model = torch.nn.Sequential(*layer_list)

        return model

    def configure_optimizers(self):
        """https://pytorch.org/docs/stable/optim.html"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._l2_scale
            )
        return optimizer

    def forward(self, x):
        """Return probabilities."""
        return F.softmax(self.forward_train(x))

    def forward_train(self, x):
        """Return logits."""
        return self._pt_model(x)

    def training_step(self, train_batch, batch_idx):
        """Return training loss and co."""
        x, y = train_batch
        logits = self.forward_train(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits)
        return {"loss": loss, "preds": preds, "target": y}

    def training_step_end(self, outputs):
        """Update and log training metrics."""
        # self.train_acc(outputs["preds"], outputs["target"])
        # metrics = {
        #     "train_acc" : self.train_acc,
        #     "train_loss" : outputs["loss"],
        #     "step" : self.current_epoch + 1
        #     }

        # changing "step" for x-axis change
        metrics = {
            "train_loss" : outputs["loss"],
            "step" : self.current_epoch + 1
            }
        self.log_dict(metrics, on_step=False, on_epoch=True)

    def validation_step(self, val_batch, batch_idx):
        """Return validation loss and co."""
        x, y = val_batch
        logits = self.forward_train(x)
        loss = F.cross_entropy(logits, y)
        preds = F.softmax(logits)
        return {"loss": loss, "preds": preds, "target": y}

    def validation_step_end(self, outputs):
        """Update and log validation metrics."""
        # self.valid_acc(outputs["preds"], outputs["target"])
        # metrics = {
        #     "valid_acc" : self.valid_acc,
        #     "valid_loss" : outputs["loss"],
        #     "step" : self.current_epoch + 1
        #     }

        # changing "step" for x-axis change
        metrics = {
            "valid_loss" : outputs["loss"],
            "step" : self.current_epoch + 1
            }
        self.log_dict(metrics, on_step=False, on_epoch=True)

    # def validation_epoch_end(self) -> None:
    #     """Reset torchmetric metrics."""
    #     self.train_acc.reset()
    #     self.valid_acc.reset()

    def print_info_summary(self, batch_size=1):
        """Print torchinfo summary."""
        print("--MODEL SUMMARY--")
        summary(
            model=self,
            input_size=(batch_size, self._x_size),
            col_names=["input_size", "output_size", "num_params"]
            )

#TODO : use class indices instead of one-hots for cross entropy loss computation
