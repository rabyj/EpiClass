import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import pandas
import tensorflow as tf
import os.path
from scipy import signal
from abc import ABC
import math

class BaseModel(ABC):
    def __init__(self):
        self._x = None
        self._y = None
        self._x_size = None
        self._y_size = None
        self._keep_prob = tf.placeholder(tf.float32)
        self._l1_scale = tf.placeholder(tf.float32)
        self._l2_scale = tf.placeholder(tf.float32)
        self._learning_rate = tf.placeholder(tf.float32)
        self._is_training = tf.placeholder(tf.bool)
        self._model = None
        self._loss = None
        self._optimizer = None
        self._predictor = None
        self._layers = []
        self._preprocess = lambda x: x

    @property
    def x(self):
        return self._x

    @property
    def x_size(self):
        return self._x_size

    @property
    def y(self):
        return self._y

    @property
    def y_size(self):
        return self._y_size
    
    @property
    def keep_prob(self):
        return self._keep_prob

    @property
    def l1_scale(self):
        return self._l1_scale

    @property
    def l2_scale(self):
        return self._l2_scale

    @property
    def is_training(self):
        return self._is_training

    @property
    def learning_rate(self):
        return self._learning_rate

    @property
    def model(self):
        return self._model

    @property
    def loss(self):
        return self._loss

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def predictor(self):
        return self._predictor

    @property
    def preprocess(self):
        return self._preprocess

    @property
    def layers(self):
        return self._layers

class StandardModel(BaseModel, ABC):
    def __init__(self):
        super().__init__()

    def _init_loss(self):
        with tf.name_scope('Loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._y , logits=self._model) + tf.losses.get_regularization_loss())
        return loss

    def _init_optimizer(self):
        with tf.name_scope('Optimizer'):
            optimizer = tf.train.AdamOptimizer(self._learning_rate).minimize(self._loss)
        return optimizer

    def _init_predictor(self):
        with tf.name_scope('Predictor'):
            predictor = tf.nn.softmax(self._model)
        return predictor


class Cnn(StandardModel):
    def __init__(self, input_size, output_size, shape):
        super().__init__()
        self._x_size = input_size
        self._y_size = output_size
        self._shape = shape
        self._x = tf.placeholder(tf.float32, [None, self._x_size])
        self._y = tf.placeholder(tf.float32, [None, self._y_size])
        self._model = self._init_model()
        self._loss = self._init_loss()
        self._optimizer = self._init_optimizer()
        self._predictor = self._init_predictor()
        self._preprocess = self._init_preprocess()

    def _init_model(self):
        conv1_size = [5, 5]
        conv1_filters = 32
        pool1_size = [2, 2]

        conv2_size = [5, 5]
        conv2_filters = 64
        pool2_size = [2, 2]

        input_layer = tf.reshape(self._x, [-1, self._shape[0], self._shape[1], 1])
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=conv1_filters, kernel_size=conv1_size, padding="same", activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=pool1_size, strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=conv2_filters, kernel_size=conv2_size, padding="same", activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=pool2_size, strides=2)
        after_conv_size = int(self._shape[0]/(pool1_size[0]*pool2_size[0])) * int(self._shape[1]/(pool1_size[1]*pool2_size[1])) * conv2_filters
        pool2_flat = tf.reshape(pool2, [-1, after_conv_size])
        hl_units = int((after_conv_size + self._y_size)/2)
        dense = tf.layers.dense(inputs=pool2_flat, units=hl_units, activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
        dropout = tf.nn.dropout(dense, self._keep_prob)
        with tf.name_scope('Model'):
            model = tf.layers.dense(inputs=dropout, units=self._y_size, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
        return model

    def _init_preprocess(self):
        return lambda x: np.reshape(signal.spectrogram(x, window=("gaussian",.005), noverlap=40, nfft=80, nperseg=64, fs=16000)[2], self._x_size)


class BidirectionalRnn(StandardModel):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._x_size = input_size
        self._y_size = output_size
        self._x = tf.placeholder(tf.float32, [None, self._x_size])
        self._y = tf.placeholder(tf.float32, [None, self._y_size])
        self._model = self._init_model()
        self._loss = self._init_loss()
        self._optimizer = self._init_optimizer()
        self._predictor = self._init_predictor()

    def _init_model(self):
        seq_len = 5
        hl_units = int((self._x_size + self._y_size)/2)*2
        nb_layers = 1

        #tf_b_VCCs_AMs_BN1 = tf.layers.batch_normalization(tf_b_VCCs_AMs, # the input vector, size [#batches, #time_steps, 2]
        #axis=-1, # axis that should be normalized 
        #training=Flg_training, # Flg_training = True during training, and False during test
        #trainable=True,
        #name="Inputs_BN"
        #)

        sequences = tf.reshape(self._x, (-1, seq_len, int(self._x_size/seq_len)))
        input_layer = tf.unstack(sequences, seq_len, 1)

        layers=[input_layer]
        for i in range(nb_layers):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(int(hl_units/2), forget_bias=1.0)
            lstm_fw_dropout = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, self._keep_prob)
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(int(hl_units/2), forget_bias=1.0)
            lstm_bw_dropout = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, self._keep_prob)
            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_dropout, lstm_bw_dropout, layers[i], dtype=tf.float32)
            layers.append(outputs[-1])

        dense = tf.layers.dense(inputs=layers[-1], units=hl_units, activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
        dropout = tf.nn.dropout(dense, self._keep_prob)
        with tf.name_scope('Model'):
            model = tf.layers.dense(inputs=dropout, units=self._y_size, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
        return model



class Dense(StandardModel):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._x_size = input_size
        self._y_size = output_size
        self._x = tf.placeholder(tf.float32, [None, self._x_size])
        self._y = tf.placeholder(tf.float32, [None, self._y_size])
        self._model = self._init_model()
        self._loss = self._init_loss()
        self._optimizer = self._init_optimizer()
        self._predictor = self._init_predictor()

    def _init_model(self):
        level = 1
        hl_units = int(math.sqrt(self._x_size + self._y_size)) * level
        nb_layers=1 * level

        self.layers.append(self._x)
        for i in range(nb_layers):
            dense = tf.layers.dense(inputs=self.layers[i], units=hl_units, activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
            dropout = tf.nn.dropout(dense, self.keep_prob)
            self.layers.append(dropout)
        with tf.name_scope('Model'):
            model = tf.layers.dense(inputs=self.layers[-1], units=self._y_size, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
            self.layers.append(model)
        return model


class L1Dense(StandardModel):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._x_size = input_size
        self._y_size = output_size
        self._x = tf.placeholder(tf.float32, [None, self._x_size])
        self._y = tf.placeholder(tf.float32, [None, self._y_size])
        self._model = self._init_model()
        self._loss = self._init_loss()
        self._optimizer = self._init_optimizer()
        self._predictor = self._init_predictor()

    def _init_model(self):
        hl_units = int((self._x_size + self._y_size))
        nb_layers=3

        dense = tf.layers.dense(inputs=self._x, units=hl_units, activation=tf.nn.sigmoid, kernel_regularizer= tf.contrib.layers.l1_regularizer(scale=self._l1_scale))

        layers = [dense]
        for i in range(nb_layers-1):
            dense = tf.layers.dense(inputs=layers[i], units=hl_units, activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
            dropout = tf.nn.dropout(dense, self.keep_prob)
            layers.append(dropout)
        with tf.name_scope('Model'):
            model = tf.layers.dense(inputs=layers[-1], units=self._y_size, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
        return model


class BatchNormDense(StandardModel):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._x_size = input_size
        self._y_size = output_size
        self._x = tf.placeholder(tf.float32, [None, self._x_size])
        self._y = tf.placeholder(tf.float32, [None, self._y_size])
        self._model = self._init_model()
        self._loss = self._init_loss()
        self._optimizer = self._init_optimizer()
        self._predictor = self._init_predictor()

    def _init_model(self):
        hl_units = int(math.sqrt(self._x_size + self._y_size))
        nb_layers=1

        layers = [self._x]
        for i in range(nb_layers):
            dense = tf.layers.dense(inputs=layers[i], units=hl_units, activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
            batch_norm = tf.contrib.layers.batch_norm(dense, center=True, scale=True, is_training=self._is_training)
            dropout = tf.nn.dropout(batch_norm, self.keep_prob)
            layers.append(dropout)
        with tf.name_scope('Model'):
            model = tf.layers.dense(inputs=layers[-1], units=self._y_size, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._l2_scale))
        return model