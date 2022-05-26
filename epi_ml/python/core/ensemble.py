from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.manifold import MDS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class Ensemble(object):
    def __init__(self, data):
        self._data = data
        self._clf = RandomForestClassifier(n_estimators=1000)

    def train(self):
        self._clf = self._clf.fit(self._data.train.signals, self._data.train.encoded_labels)

    def metrics(self):
        y_pred = self._clf.predict(self._data.train.signals)
        y_true = self._data.train.encoded_labels
        print ("Test Accuracy: %s" % sklearn.metrics.accuracy_score(y_true, y_pred))

        y_pred = self._clf.predict(self._data.validation.signals)
        y_true =  self._data.validation.encoded_labels
        print ("Validation Accuracy: %s" % sklearn.metrics.accuracy_score(y_true, y_pred))
        print ("Validation Precision: %s" % sklearn.metrics.precision_score(y_true, y_pred, average="macro"))
        print ("Validation Recall: %s" % sklearn.metrics.recall_score(y_true, y_pred, average="macro"))
        print ("Validation f1_score: %s" % sklearn.metrics.f1_score(y_true, y_pred, average="macro"))
