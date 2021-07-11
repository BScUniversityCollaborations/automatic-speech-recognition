from datetime import date

import librosa
import scipy.signal as sg
from matplotlib import pyplot as plt
from scipy.io import wavfile


def remove_noise(file_path, signal, sample_rate):
    # Butterworth filter
    n = 1  # Filter order
    wn = 0.15  # Cutoff frequency
    btype, analog = sg.butter(n, wn, output='ba')

    # Applying the filter
    signal_reduced_noise = sg.filtfilt(btype, analog, signal)

    # sample_rate, samples = wavfile.read(file_path)
    frequencies, times, spectrogram = sg.spectrogram(signal, sample_rate)

    plt.pcolormesh(times, frequencies, spectrogram)
    plt.imshow(spectrogram)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

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

