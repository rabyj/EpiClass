import matplotlib

matplotlib.use("Agg")

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def spectro_hg(path):
    f = h5py.File(path)
    for group in f:
        md5 = group
        break
    array = np.array([])
    # fmt: off
    for i in ["chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10","chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19","chr20","chr21","chr22","chrX"]:  # fmt: on
        array = np.concatenate((array, f[md5][i][...]), axis=0)
    norm = (array - array.mean()) / array.std()
    # print(norm.size)
    spectro(norm)


def spectro(x):
    f, t, Sxx = signal.spectrogram(
        x, window=("tukey", 0.005), noverlap=40, nfft=80, nperseg=64, fs=16000
    )
    # f, t, Sxx = signal.spectrogram(x, window=("gaussian",5), nperseg=2**16)
    # f, t, Sxx = signal.spectrogram(x)
    # plt.figure()
    # plt.plot(x)
    plt.figure()
    plt.pcolormesh(t, f, np.log10(Sxx))
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [sec]")
    plt.show()
