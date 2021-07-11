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

    # sample_rate, samples = wavfile.read(file_path)
    # frequencies, times, spectrogram = sg.spectrogram(signal, sample_rate)
    #
    # plt.pcolormesh(times, frequencies, spectrogram)
    # plt.imshow(spectrogram)
    # plt.ylabel('Frequency [Hz]')
    # plt.xlabel('Time [sec]')
    # plt.show()

    # Make plots
    # plt.subplot(211)
    # plt.title('Spectrogram of audio file')
    # plt.legend(['Original', 'Filtered Noise'])
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    #
    # plt.subplot(212)
    # plt.specgram(signal_reduced_noise, Fs=freq_sampling)
    # plt.xlabel('Time')
    # plt.ylabel('Frequency')
    # plt.show()

    return signal_reduced_noise


def show_plots(signal_data, signal_data_reduced, sample_rate):
    plt.figure(1)
    plt.title('Waveform')
    librosa.display.waveshow(signal_data_reduced, sr=sample_rate)
    plt.show()
