from abc import ABC, abstractmethod

import sklearn.metrics
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from .analysis import write_pred_table
from .data import DataSet


class AbstractEstimator(ABC):
    """Generic abstract estimator class"""
    @abstractmethod
    def __init__(self, data: DataSet):
        self._data = data
        self._clf = None
        self._results = None
        self._mapping = dict(enumerate(data.classes))

    def train(self):
        """Fit model to training data."""
        self._clf = self._clf.fit(self._data.train.signals, self._data.train.encoded_labels)

    def metrics(self):
        """Print metrics"""
        y_pred = self._clf.predict(self._data.train.signals)
        y_true = self._data.train.encoded_labels
        print (f"Test Accuracy: {sklearn.metrics.accuracy_score(y_true, y_pred)}")

        y_pred = self._clf.predict(self._data.validation.signals)
        y_true = self._data.validation.encoded_labels
        print (f"Validation Accuracy: {sklearn.metrics.accuracy_score(y_true, y_pred)}")
        print (f"Validation Precision: {sklearn.metrics.precision_score(y_true, y_pred, average='macro')}")
        print (f"Validation Recall: {sklearn.metrics.recall_score(y_true, y_pred, average='macro')}")
        print (f"Validation f1_score: {sklearn.metrics.f1_score(y_true, y_pred, average='macro')}")

        self._results = y_pred

    def predict_file(self, log):
        """Write pred table"""
        if self._results is None:
            self._results = self._clf.predict(self._data.validation.signals)

        str_preds = [self._mapping[encoded_label] for encoded_label in self._results]

        write_pred_table(
            predictions=self._results,
            str_preds=str_preds,
            str_targets=self._data.validation.original_labels,
            classes=self._data.classes,
            md5s=self._data.validation.ids,
            path=log
        )


class Ensemble(AbstractEstimator):
    """A simple Random Forest classifier."""
    def __init__(self, data):
        super().__init__(data)
        self._clf = RandomForestClassifier(n_estimators=1000)


class Svm(AbstractEstimator):
    """A simple SVM classifier."""
    def __init__(self, data):
        super().__init__(data)
        self._clf = svm.SVC()
