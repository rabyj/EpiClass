"""Module for wrappers around simple sklearn machine learning estimators."""
import sklearn.metrics
from sklearn.preprocessing import LabelBinarizer

from .analysis import write_pred_table


class EstimatorAnalyzer(object):
    """Generic class to analyze results given by an estimator."""

    def __init__(self, classes, estimator):
        self.classes = sorted(classes)
        self.mapping = dict(enumerate(self.classes))
        self._clf = estimator
        self.encoder = LabelBinarizer().fit(list(self.mapping.keys()))

    def metrics(self, X, y, verbose=True):
        """Return a dict of metrics over given set"""
        y_pred = self._clf.predict(X)
        y_true = y

        val_acc = sklearn.metrics.accuracy_score(y_true, y_pred)
        val_precision = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
        val_recall = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
        val_f1 = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
        val_mcc = sklearn.metrics.matthews_corrcoef(y_true, y_pred)

        metrics_dict = {
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1,
            "val_mcc": val_mcc,
        }

        if verbose:
            EstimatorAnalyzer.print_metrics(metrics_dict)

        return metrics_dict

    @staticmethod
    def print_metrics(metrics_dict: dict):
        """Print metrics"""
        print(f"Validation Accuracy: {metrics_dict['val_acc']}")
        print(f"Validation Precision: {metrics_dict['val_precision']}")
        print(f"Validation Recall: {metrics_dict['val_recall']}")
        print(f"Validation F1_score: {metrics_dict['val_f1']}")
        print(f"Validation MCC: {metrics_dict['val_mcc']}")

    def predict_file(self, ids, X, y, log):
        """Write predictions table for validation set."""

        int_results = self._clf.predict(X)
        pred_results = self.encoder.transform(int_results)

        str_preds = [self.mapping[encoded_label] for encoded_label in int_results]
        str_y = [self.mapping[encoded_label] for encoded_label in y]

        write_pred_table(
            predictions=pred_results,
            str_preds=str_preds,
            str_targets=str_y,
            classes=self.classes,
            md5s=ids,
            path=log,
        )
