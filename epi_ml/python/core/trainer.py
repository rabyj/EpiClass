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


class Trainer(object):
    def __init__(self, data, model, logdir):
        self._data = data
        self._model = model
        self._logdir = logdir
        self._data.preprocess(model.preprocess)
        self._hparams = {
            "learning_rate": 1e-4,
            "training_epochs": 100,
            "batch_size": 512,
            "measure_frequency": 1,
            "l1_scale": 0.001,
            "l2_scale": 0.01,
            "keep_prob": 0.5,
            "is_training": True
        }
        self._train_accuracy = self._init_accuracy("Training_Accuracy")
        self._valid_accuracy = self._init_accuracy("Validation_Accuracy")
        self._test_accuracy = self._init_accuracy("Test_Accuracy")
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
        sess.run(tf.global_variables_initializer())
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

    def train(self):
        #train
        nb_batch = math.ceil(self._data.train.num_examples/self._hparams.get("batch_size"))
        for epoch in range(self._hparams.get("training_epochs")):
            for batch in range(nb_batch):
                batch_xs, batch_ys = self._data.train.next_batch(self._hparams.get("batch_size"))
                if epoch % self._hparams.get("measure_frequency") == 0 and batch == 0:
                    t_acc = self._sess.run(self._train_accuracy, feed_dict=self._make_dict(batch_xs, batch_ys, keep_prob=1.0))
                    v_acc, v_summary = self._sess.run([self._valid_accuracy, self._v_sum], feed_dict=self._make_dict(self._data.validation.signals, self._data.validation.labels, keep_prob=1.0, is_training=False))
                    self._writer.add_summary(v_summary, epoch)
                    print('epoch {0}, training accuracy {1:.4f}, validation accuracy {2:.4f}'.format(epoch, t_acc, v_acc))
                _, _, summary = self._sess.run([self._model.optimizer, self._model.loss, self._summary], feed_dict=self._make_dict(batch_xs, batch_ys))
                self._writer.add_summary(summary, epoch)

    def metrics(self):
        test_acc, pred = self._sess.run([self._test_accuracy, self._model.predictor], feed_dict=self._make_dict(self._data.test.signals, self._data.test.labels, keep_prob=1.0, is_training=False))
        print("Accuracy: %s" % (test_acc))
        y_true = np.argmax(self._data.test.labels,1)
        y_pred = np.argmax(pred,1)
        print ("Precision: %s" % sklearn.metrics.precision_score(y_true, y_pred, average="macro"))
        print ("Recall: %s" % sklearn.metrics.recall_score(y_true, y_pred, average="macro"))
        print ("f1_score: %s" % sklearn.metrics.f1_score(y_true, y_pred, average="macro"))
        self.write_pred_table(pred, self._data.labels, self._data.test.labels)
        self.heatmap(self._data.labels)

    def visualize(self, vis):
        outputs = self._sess.run(self._model.layers, feed_dict=self._make_dict(self._data.train.signals, self._data.train.labels, keep_prob=1.0, is_training=False))

        for idx, output in enumerate(outputs):
            vis.run(output, self._data.train.labels, self._sess, self._writer, str(idx))

    def importance(self):
        #garson algorithm #TODO: generalise, put in model
        w = self._sess.run(tf.trainable_variables())
        total_w = w[0]
        for i in range(2, len(w), 2):
            total_w = np.dot(total_w, w[i])
        total_w = np.absolute(total_w)
        sum_w = np.sum(total_w, axis=None)
        total_w = np.sum(total_w/sum_w, axis=1)
        print((total_w > 1e-04).sum())
        return ','.join([str(x) for x in total_w])

    def heatmap(self, labels):
        confusion_mat = tf.confusion_matrix(tf.argmax(self._model.model,1), tf.argmax(self._model.y,1))
        confusion_matrix = self._sess.run(confusion_mat, feed_dict=self._make_dict(self._data.test.signals, self._data.test.labels, keep_prob=1.0, is_training=False))
        plt.figure()
        confusion_matrix[(confusion_matrix > 0)] = 1

        fig, ax = plt.subplots()
        ax.pcolor(confusion_matrix, cmap=plt.cm.Blues, alpha=0.8)
        ax.set_frame_on(False)
        ax.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
        ax.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
        ax.invert_yaxis()
        ax.xaxis.tick_top()
        ax.set_xticklabels(labels, fontsize=2)
        ax.set_yticklabels(labels, fontsize=2)
        plt.xticks(rotation=90)
        ax = plt.gca()
        for t in ax.xaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False
        for t in ax.yaxis.get_major_ticks():
            t.tick1On = False
            t.tick2On = False

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=400)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        summary = tf.summary.image("Confusion Matrix", image, max_outputs=1)
        self._writer.add_summary(summary.eval(session=self._sess))

    def write_pred_table(self, pred, pred_labels, labels):
        string_labels = [pred_labels[np.argmax(label)] for label in labels]
        df = pandas.DataFrame(data=pred, index=string_labels, columns=pred_labels)
        pred_str = df.to_csv(os.path.join(self._logdir, "predict.csv"), encoding="utf8")
