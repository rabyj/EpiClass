from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics
from sklearn.manifold import MDS, TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import io
import tensorflow as tf
from abc import ABC
import itertools
import collections

class Visualization(ABC):
    KELLY_COLORS = [(1.0, 0.7019607843137254, 0.0, 1.),
                    (0.5019607843137255, 0.24313725490196078, 0.4588235294117647, 1.),
                    (1.0, 0.40784313725490196, 0.0, 1.),
                    (0.6509803921568628, 0.7411764705882353, 0.8431372549019608, 1.),
                    (0.7568627450980392, 0.0, 0.12549019607843137, 1.),
                    (0.807843137254902, 0.6352941176470588, 0.3843137254901961, 1.),
                    (0.5058823529411764, 0.4392156862745098, 0.4, 1.),
                    (0.0, 0.49019607843137253, 0.20392156862745098, 1.),
                    (0.9647058823529412, 0.4627450980392157, 0.5568627450980392, 1.),
                    (0.0, 0.3254901960784314, 0.5411764705882353, 1.),
                    (1.0, 0.47843137254901963, 0.3607843137254902, 1.),
                    (0.3254901960784314, 0.21568627450980393, 0.47843137254901963, 1.),
                    (1.0, 0.5568627450980392, 0.0, 1.),
                    (0.7019607843137254, 0.1568627450980392, 0.3176470588235294, 1.),
                    (0.9568627450980393, 0.7843137254901961, 0.0, 1.),
                    (0.4980392156862745, 0.09411764705882353, 0.050980392156862744, 1.),
                    (0.5764705882352941, 0.6666666666666666, 0.0, 1.),
                    (0.34901960784313724, 0.2, 0.08235294117647059, 1.),
                    (0.9450980392156862, 0.22745098039215686, 0.07450980392156863, 1.),
                    (0.13725490196078433, 0.17254901960784313, 0.08627450980392157, 1.)]

    def __init__(self):
        self._summary_name = 'Visualization'

    def color(self, y):
        colors_palette = itertools.chain(self.KELLY_COLORS, itertools.cycle([(0., 0., 0., 0.05)]))
        labels = [str(label) for label in y]
        ordered_label_set = [x[0] for x in collections.Counter(labels).most_common()]
        color_chart = {label: n for label, n in zip(ordered_label_set, colors_palette)}
        colors = [color_chart[label] for label in labels]
        return colors

    def show(self, y, sess, writer, summary_name=None):
        if summary_name is None:
            summary_name = self._summary_name

        plt.figure()
        plt.scatter(*zip(*self._embedding), c=self.color(y))

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=400)
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        summary = tf.summary.image(summary_name, image, max_outputs=1)        
        writer.add_summary(summary.eval(session=sess))

    def train(self, x, y):
        self._embedding = self._clf.fit_transform(x, y)

    def run(self, x, y, sess, writer, summary_name=None):
        self.train(x, y)
        self.show(y, sess, writer, summary_name)

class Mds(Visualization):
    def __init__(self):
        super().__init__()
        self._summary_name = 'Mds'
        self._clf = MDS()

class Tsne(Visualization):
    def __init__(self):
        super().__init__()
        self._summary_name = 'Tsne'
        self._clf = TSNE()

class Pca(Visualization):
    def __init__(self):
        super().__init__()
        self._summary_name = 'Pca'
        self._clf = PCA(2)
        