import torch
from torch import nn
import pytorch_lightning as pl

class LightningDenseClassifier(pl.LightningModule):

    def __init__(self, input_size, output_size, hparams, hl_units=3000, nb_layer=1):
        super(LightningEpiClassifier, self).__init__()

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
        for i in range(self._nb_layer - 1):
            layer_list.append(torch.nn.Linear(self._hl_size, self._hl_size))
            # in case of ReLU, dropout should be applied before for computational efficiency https://sebastianraschka.com/faq/docs/dropout-activation.html
            layer_list.append(torch.nn.ReLU())
            layer_list.append(torch.nn.Dropout(self._dropout_rate))

        # output layer
        layer_list.append(torch.nn.Linear(self._hl_size, self._y_size))

        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/container.html#Sequential
        model = torch.nn.Sequential(*layer_list)

        return model

# TO ADAPT
# model.add(tf.keras.Input(shape=(self._x_size,)))

# for i in range(self._nb_layer):
#     model.add(tf.keras.layers.Dense(
#         units=self._hl_size,
#         activation="relu",
#         kernel_regularizer=tf.keras.regularizers.L2(l2=self._l2_scale),
#         name="dense_{}".format(i)
#         ))
#     model.add(tf.keras.layers.Dropout(self._dropout_rate))


    def configure_optimizers(self):
        """https://pytorch.org/docs/stable/optim.html"""
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self._learning_rate,
            weight_decay=self._l2_scale
            )
        return optimizer
