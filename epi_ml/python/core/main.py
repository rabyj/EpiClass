import os
import sys
import warnings
warnings.simplefilter("ignore")
import data
import model
import trainer
import config
import os.path
import matplotlib.pyplot as plt
from scipy import signal

def spectro(x):
        f, t, Sxx = signal.spectrogram(x, window=("gaussian",.005), noverlap=40, nfft=80, nperseg=64, fs=16000)
        print(Sxx.shape)
        plt.figure()
        plt.pcolormesh(t, f, Sxx)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()

def main(args):
    my_data = data.EpiData("assay", oversample=False)
    #my_data = data.EpiData("publishing_group")

    #spectro(my_data.test.signals[55])

    input_size = my_data.train.signals[0].size
    ouput_size = my_data.train.labels[0].size

    my_model = model.Dense(input_size, ouput_size)
    #my_model = model.Cnn(41*49, ouput_size, (41, 49))
    #my_model = model.BidirectionalRnn(input_size, ouput_size)

    my_trainer = trainer.Trainer(my_data, my_model)
    my_trainer.train()
    my_trainer.metrics()

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    main(sys.argv[1:])
