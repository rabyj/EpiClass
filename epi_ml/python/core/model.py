import io
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import pandas
import tensorflow as tf
import os.path
from scipy import signal
from abc import ABC

import config

class Model(ABC):
    def __init__(self):
        self._x = None
        self._y = None
        self._x_size = None
        self._y_size = None
        self._keep_prob = tf.placeholder(tf.float32)
        self._beta = tf.placeholder(tf.float32)
        self._learning_rate = tf.placeholder(tf.float32)
        self._model = None
        self._loss = None
        self._optimizer = None
        self._predictor = None

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
    def beta(self):
        return self._beta

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


class Cnn(Model):
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
        input_layer = tf.reshape(self._x, [-1, 129, 5, 1])
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 32 * 1 * 64])
        hl_units = 1024
        dense = tf.layers.dense(inputs=pool2_flat, units=hl_units, activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._beta))
        dropout = tf.nn.dropout(dense, self._keep_prob)
        with tf.name_scope('Model'):
            model = tf.layers.dense(inputs=self._model.output, units=self._model.size_y, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._hyperparams.get("beta")))
        return model

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


class BidirectionalRnn(Model):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._x_size = input_size
        self._y_size = output_size
        self._x = tf.placeholder(tf.float32, [None, 5, self._x_size/5])
        self._y = tf.placeholder(tf.float32, [None, output_size])
        self._model = self._init_model()
        self._loss = self._init_loss()
        self._optimizer = self._init_optimizer()
        self._predictor = self._init_predictor()

    def _init_model(self):
        input_layer = tf.unstack(self._x, 5, 1)
        hl_units = int((self._x_size + self._y_size)/2)*2
        lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(int(hl_units/2), forget_bias=1.0)
        lstm_fw_dropout = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, self._keep_prob)
        lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(int(hl_units/2), forget_bias=1.0)
        lstm_bw_dropout = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, self._keep_prob)
        outputs, a, b = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_dropout, lstm_bw_dropout, input_layer, dtype=tf.float32)
        dense = tf.layers.dense(inputs=outputs[-1], units=hl_units, activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._beta))
        dropout = tf.nn.dropout(dense, self._keep_prob)
        with tf.name_scope('Model'):
            model = tf.layers.dense(inputs=self._model.output, units=self._model.size_y, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._hyperparams.get("beta")))
        return model

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


class Dense(Model):
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
        hl_units = int((self._x_size + self._y_size)/2)
        dense = tf.layers.dense(inputs=self._x, units=hl_units, activation=tf.nn.relu, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._beta))
        dropout = tf.nn.dropout(dense, self.keep_prob)
        with tf.name_scope('Model'):
            model = tf.layers.dense(inputs=dropout, units=self._y_size, kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self._beta))
        return model

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
