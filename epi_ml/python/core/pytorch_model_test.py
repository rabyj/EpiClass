import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class LightningDenseClassifier(pl.LightningModule):

    def __init__(self, input_size, output_size, hparams, hl_units=3000, nb_layer=1):
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

        # self.layer_1 = torch.nn.Linear(input_size, hl_units)
        # self.layer_3 = torch.nn.Linear(hl_units, output_size)

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
        """Return probabilities"""
        return F.softmax(self._forward_train(x))

    def forward_train(self, x):
        """Return logits"""
        return self._pt_model(x)

    def training_step(self, train_batch, batch_idx):
        "Do things each step."
        x, y = train_batch
        logits = self.forward_train(x) # do not call forward directly?
        loss = F.cross_entropy(logits, y)
        # changing "step" for x-axis change
        metrics = {
            "train_loss_epoch" : loss,
            "step" : self.current_epoch + 1
            }
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward_train(x)
        loss = F.cross_entropy(logits, y)
        metrics = {
            "valid_loss_epoch" : loss,
            "step" : self.current_epoch + 1
            }
        self.log_dict(metrics, on_step=False, on_epoch=True)

#TODO : add accuracy to validation step. Use old method or torchmetrics?


# class LightningMNISTClassifier(pl.LightningModule):

#   def __init__(self):
#     super().__init__()

#     # mnist images are (1, 28, 28) (channels, width, height)
#     self.layer_1 = torch.nn.Linear(28 * 28, 128)
#     self.layer_2 = torch.nn.Linear(128, 256)
#     self.layer_3 = torch.nn.Linear(256, 10)

#   def forward(self, x):
#       batch_size, channels, width, height = x.size()

#       # (b, 1, 28, 28) -> (b, 1*28*28)
#       x = x.view(batch_size, -1)

#       # layer 1 (b, 1*28*28) -> (b, 128)
#       x = self.layer_1(x)
#       x = torch.relu(x)

#       # layer 2 (b, 128) -> (b, 256)
#       x = self.layer_2(x)
#       x = torch.relu(x)

#       # layer 3 (b, 256) -> (b, 10)
#       x = self.layer_3(x)

#       # probability distribution over labels
#       x = torch.log_softmax(x, dim=1)

#       return x

#   def forward_train(self, x):
#       batch_size, channels, width, height = x.size()

#       # (b, 1, 28, 28) -> (b, 1*28*28)
#       x = x.view(batch_size, -1)

#       # layer 1 (b, 1*28*28) -> (b, 128)
#       x = self.layer_1(x)
#       x = torch.relu(x)

#       # layer 2 (b, 128) -> (b, 256)
#       x = self.layer_2(x)
#       x = torch.relu(x)

#       # layer 3 (b, 256) -> (b, 10)
#       x = self.layer_3(x)

        ### NO SOFTMAX ###

#       return x

