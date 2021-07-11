import librosa
import librosa.display
import numpy as np
import scipy.signal as sg
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks


def remove_noise(signal_data, sample_rate):
    # Butterworth filter
    n = 1  # Filter order
    wn = 0.15  # Cutoff frequency
    btype, analog = sg.butter(n, wn, output='ba')

    # Applying the filter
    signal_reduced_noise = sg.filtfilt(btype, analog, signal_data)

    show_plots(signal_data, signal_reduced_noise, sample_rate)

    return signal_reduced_noise


def show_plots(signal_data, signal_data_reduced, sample_rate):
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex="all", constrained_layout=True)

    ax.set(title="Original Audio Waveform Graph", xlabel="Time", ylabel="Amplitude")
    ax2.set(title="Audio Waveform Graph", xlabel="Time", ylabel="Amplitude")

    ax.grid()
    ax2.grid()

    librosa.display.waveshow(signal_data, sr=sample_rate, ax=ax, label="Original")
    librosa.display.waveshow(signal_data, sr=sample_rate, ax=ax2, label="Original")
    librosa.display.waveshow(signal_data_reduced, sr=sample_rate, ax=ax2, label="Filtered")

    ax2.legend()

    plt.show()

    # fig.savefig("test.png")
