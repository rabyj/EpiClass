from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class Mds(object):
    def __init__(self, data):
        self._data = data
        self._clf = MDS()

    def train(self):
        self._embedding = self._clf.fit_transform(self._data.train.signals, self._data.train.labels)

    def color(self):
        label_set = set(self._data.train.labels)
        color_chart = {label: n for label, n in zip(label_set, range(len(label_set)))}
        colors = [color_chart[label] for label in self._data.train.labels]
        return colors

    def show(self):
        plt.figure()
        plt.scatter(*zip(*self._embedding), c=self.color())
        plt.show()

class Tsne(object):
    def __init__(self, data):
        self._data = data
        self._clf = TSNE()

    def train(self):
        self._embedding = self._clf.fit_transform(self._data.train.signals, self._data.train.labels)

    def color(self):
        label_set = set(self._data.train.labels)
        color_chart = {label: n for label, n in zip(label_set, range(len(label_set)))}
        colors = [color_chart[label] for label in self._data.train.labels]
        return colors

    def show(self):
        plt.figure()
        plt.scatter(*zip(*self._embedding), c=self.color())
        plt.show()

class Pca(object):
    def __init__(self, data):
        self._data = data
        self._clf = PCA(2)

    def train(self):
        self._embedding = self._clf.fit_transform(self._data.train.signals, self._data.train.labels)

    def color(self):
        label_set = set(self._data.train.labels)
        color_chart = {label: n for label, n in zip(label_set, range(len(label_set)))}
        colors = [color_chart[label] for label in self._data.train.labels]
        return colors

    def show(self):
        plt.figure()
        plt.scatter(*zip(*self._embedding), c=self.color())
        plt.show()
        