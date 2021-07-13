import librosa
import librosa.display
import numpy as np
import scipy.signal as sg
from matplotlib import pyplot as plt
from numpy.lib import stride_tricks
from constants import *


def show_plots_compare_two_signals(signal_data, signal_data_reduced):
    fig, ax = plt.subplots(nrows=2, sharex="all", sharey="all", constrained_layout=True)

    ax[0].set(title="Original Audio Waveform Graph", xlabel=TXT_TIME, ylabel=TXT_AMPLITUDE)
    ax[1].set(title="Audio Waveform Graph", xlabel=TXT_TIME, ylabel=TXT_AMPLITUDE)

    # Apply grid
    ax[0].grid()
    ax[1].grid()

    librosa.display.waveshow(signal_data, sr=DEFAULT_SAMPLE_RATE, ax=ax[0], label=TXT_ORIGINAL)
    librosa.display.waveshow(signal_data, sr=DEFAULT_SAMPLE_RATE, ax=ax[1], label=TXT_ORIGINAL)
    librosa.display.waveshow(signal_data_reduced, sr=DEFAULT_SAMPLE_RATE, ax=ax[1], label=TXT_FILTERED)

    # Set legend
    ax[1].legend()

    # Show plot
    plt.show()

    # fig.savefig("test.png")


def show_plot_emphasized(signal_data_orig, signal_data_emphasized):
    s_orig = librosa.amplitude_to_db(np.abs(librosa.stft(signal_data_orig)), ref=np.max, top_db=None)
    s_pre_emphasized = librosa.amplitude_to_db(np.abs(librosa.stft(signal_data_emphasized)), ref=np.max, top_db=None)

    fig, ax = plt.subplots(nrows=2, sharex="all", sharey="all", constrained_layout=True)
    librosa.display.specshow(s_orig, y_axis='log', x_axis='time', ax=ax[0])

    img = librosa.display.specshow(s_pre_emphasized, y_axis='log', x_axis='time', ax=ax[1])
    fig.colorbar(img, ax=ax, format="%+2.f dB")

    ax[0].label_outer()

    # Set title
    ax[0].set(title=TXT_ORIGINAL_SIGNAL, xlabel=TXT_TIME, ylabel=TXT_FREQUENCY)
    ax[1].set(title=TXT_PRE_EMPHASIZED_SIGNAL, xlabel=TXT_TIME, ylabel=TXT_FREQUENCY)

    # Show plot
    plt.show()


def show_plot_zcr(signal_data_zcr):
    plt.plot(signal_data_zcr[0])

    # Set title
    plt.title(TXT_ZERO_CROSSING_RATE)
    # Apply grid
    plt.grid()
    # Zooming in
    plt.figure(figsize=(14, 5))

    # Show plot
    plt.show()

