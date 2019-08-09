import numpy as np
import pandas
import tensorflow as tf
import os.path
from scipy import signal
from abc import ABC
import math
import datetime

from .data import DataSet


class Trainer(object):
    def __init__(self, data: DataSet, model, logdir, **kwargs):
        self._data = data
        self._model = model
        self._logdir = logdir
        self._data.preprocess(model.preprocess)
        self._hparams = {
            "learning_rate": kwargs.get("learning_rate", 1e-5),
            "training_epochs": kwargs.get("training_epochs", 50),
            "batch_size": kwargs.get("batch_size", 64),
            "measure_frequency": kwargs.get("measure_frequency", 1),
            "l1_scale": kwargs.get("l1_scale", 0.001),
            "l2_scale": kwargs.get("l2_scale", 0.01),
            "keep_prob": kwargs.get("keep_prob", 0.5),
            "is_training": kwargs.get("is_training", True),
            "early_stop_limit": kwargs.get("early_stop_limit", 15)
        }
        self._print_hparams()
        self._train_accuracy = self._init_accuracy("Training_Accuracy")
        self._valid_accuracy = self._init_accuracy("Validation_Accuracy")
        self._test_accuracy = self._init_accuracy("Test_Accuracy")
        self._run_metadata = tf.RunMetadata()
        self._run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, report_tensor_allocations_upon_oom=True)
        self._writer = self._init_writer()
        self._summary = self._init_summary()
        self._v_sum = self._init_v_sum()
        self._sess = self._start_sess()

    def __del__(self):
        if hasattr(self, "_writer"):
            self._writer.close()
        if hasattr(self, "_sess"):
            self._sess.close()

    def _start_sess(self):
        sess = tf.Session()
        sess.run(tf.global_variables_initializer(), options=self._run_options, run_metadata=self._run_metadata)
        self._writer.add_run_metadata(self._run_metadata, "init")
        return sess

    def _init_writer(self):
        return tf.summary.FileWriter(self._logdir, graph=tf.get_default_graph())

    def _init_summary(self):
        l_sum = tf.summary.scalar("loss", self._model.loss)
        t_sum = tf.summary.scalar("training_accuracy", self._train_accuracy)
        return tf.summary.merge([l_sum, t_sum])

    def _init_v_sum(self):
        return tf.summary.scalar("validation_accuracy", self._valid_accuracy)

    def _init_accuracy(self, name):
        with tf.name_scope(name):
            accuracy = tf.equal(tf.argmax(self._model.model, 1), tf.argmax(self._model.y, 1))
            accuracy = tf.reduce_mean(tf.cast(accuracy, tf.float32))
        return accuracy

    def _make_dict(self, signals, labels, keep_prob=None, l1_scale=None, l2_scale=None, learning_rate=None, is_training=None):
        if keep_prob is None:
            keep_prob = self._hparams.get("keep_prob")
        if l1_scale is None:
            l1_scale = self._hparams.get("l1_scale")
        if l2_scale is None:
            l2_scale = self._hparams.get("l2_scale")
        if learning_rate is None:
            learning_rate = self._hparams.get("learning_rate")
        if is_training is None:
            is_training = self._hparams.get("is_training")
        default_dict = {
            self._model.x: signals,
            self._model.y: labels,
            self._model.keep_prob: keep_prob,
            self._model.l1_scale: l1_scale,
            self._model.l2_scale: l2_scale,
            self._model.learning_rate: learning_rate,
            self._model.is_training: is_training
        }
        return default_dict

    def _print_hparams(self):
        for hparam, value in sorted(self._hparams.items()):
            print("{}: {}".format(hparam, value))

    def restore(self, name="save"):
        """Restore saved model."""
        saver = tf.train.Saver()
        save_path = os.path.join(self._logdir, name)
        saver.restore(self._sess, save_path)

    def train(self):
        nb_batch = math.ceil(self._data.train.num_examples/self._hparams.get("batch_size"))

        saver = tf.train.Saver(max_to_keep=1)
        save_path = os.path.join(self._logdir, "save")
        max_v_acc = -1
        nb_since_max = 0

        for epoch in range(self._hparams.get("training_epochs")):
            for batch in range(nb_batch):
                batch_xs, batch_ys = self._data.train.next_batch(self._hparams.get("batch_size"))
                if epoch % self._hparams.get("measure_frequency") == 0 and batch == 0:

                    # train
                    _, _, summary = self._sess.run([self._model.minimize, self._model.loss, self._summary], feed_dict=self._make_dict(batch_xs, batch_ys), run_metadata=self._run_metadata, options=self._run_options)
                    self._writer.add_summary(summary, epoch)

                    # batch training accuracy
                    t_acc = self._sess.run(self._train_accuracy, feed_dict=self._make_dict(batch_xs, batch_ys, keep_prob=1.0))

                    # validation accuracy
                    v_acc, v_summary = self._sess.run([self._valid_accuracy, self._v_sum], feed_dict=self._make_dict(self._data.validation.signals, self._data.validation.labels, keep_prob=1.0, is_training=False))
                    self._writer.add_summary(v_summary, epoch)

                    self._writer.add_run_metadata(self._run_metadata, "epoch{}".format(epoch))

                    if v_acc > max_v_acc:
                        max_v_acc = v_acc
                        saver.save(self._sess, save_path)
                        nb_since_max = 0
                    else:
                        nb_since_max += 1

                    if nb_since_max == self._hparams.get("early_stop_limit"):
                        break
                    
                    print("epoch {0}, batch training accuracy {1:.4f}, validation accuracy {2:.4f} {3}".format(epoch, t_acc, v_acc, datetime.datetime.now()))

                else:
                    # train
                    _, _, summary = self._sess.run([self._model.minimize, self._model.loss, self._summary], feed_dict=self._make_dict(batch_xs, batch_ys))
                    self._writer.add_summary(summary, epoch)
                
            if nb_since_max == self._hparams.get("early_stop_limit"):
                break

        # load best model
        saver.restore(self._sess, save_path)

    def _compute_acc(self, set_accuracy, data_subset):
        return self._sess.run(set_accuracy, feed_dict=self._make_dict(data_subset.signals, data_subset.labels, keep_prob=1.0, is_training=False))

    def training_acc(self):
        return self._compute_acc(self._train_accuracy, self._data.train)

    def validation_acc(self):
        return self._compute_acc(self._valid_accuracy, self._data.validation)

    def test_acc(self):
        return self._compute_acc(self._test_accuracy, self._data.test)

    def _compute_pred(self, data_subset):
        return self._sess.run(self._model.predictor, feed_dict=self._make_dict(data_subset.signals, data_subset.labels, keep_prob=1.0, is_training=False))

    def training_pred(self):
        return self._compute_pred(self._data.train)

    def validation_pred(self):
        return self._compute_pred(self._data.validation)

    def test_pred(self):
        return self._compute_pred(self._data.test)

    def _compute_conf_mat(self, data_subset):
        confusion_mat = tf.confusion_matrix(tf.argmax(self._model.model,1), tf.argmax(self._model.y,1))
        return self._sess.run(confusion_mat, feed_dict=self._make_dict(data_subset.signals, data_subset.labels, keep_prob=1.0, is_training=False))

    def training_mat(self):
        return self._compute_conf_mat(self._data.train)

    def validation_mat(self):
        return self._compute_conf_mat(self._data.validation)

    def test_mat(self):
        return self._compute_conf_mat(self._data.test)

    def weights(self):
        return self._sess.run(tf.trainable_variables())

    def visualize(self, vis):
        #TODO: maybe send to analysis module
        outputs = self._sess.run(self._model.layers, feed_dict=self._make_dict(self._data.train.signals, self._data.train.labels, keep_prob=1.0, is_training=False))

        for idx, output in enumerate(outputs):
            vis.run(output, self._data.train.labels, self._sess, self._writer, str(idx))

