import librosa.display
import scipy
import soundfile as sf
import scipy.signal as sg

from constants import *
from plots import *


def pre_processing(signal_data):
    # === Pre-Emphasis ===
    signal_emphasized = librosa.effects.preemphasis(signal_data)

    # === Filtering ===
    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_trimmed, i = librosa.effects.trim(signal_reduced_noise, TOP_DB)

    signal_zcr = librosa.feature.zero_crossing_rate(signal_trimmed)
    zcr_average = np.mean(signal_zcr)

    signal_short_time_energy = calculate_short_time_energy(signal_trimmed)

    # Show plots
    show_plot_emphasized(signal_data, signal_emphasized)
    show_plots_compare_two_signals(signal_data, signal_reduced_noise)
    show_plot_zcr(signal_zcr)
    show_plot_short_time_energy(signal_trimmed, signal_short_time_energy)

    # Exporting the filtered audio file.
    sf.write(".\\data\\samples\\sample_filtered.wav", signal_trimmed, DEFAULT_SAMPLE_RATE)

    # Print statistics
    print(TXT_LINE, "\n")
    print(TXT_STATISTICS)
    print(TXT_ORIGINAL_AUDIO_SAMPLE_RATE.format(DEFAULT_SAMPLE_RATE))
    print(TXT_ORIGINAL_AUDIO_DURATION_FORMAT.format(librosa.get_duration(signal_data)))
    print(TXT_TRIMMED_AUDIO_DURATION_FORMAT.format(librosa.get_duration(signal_trimmed)))
    print(TXT_ZCR_AVERAGE.format(zcr_average), "\n")
    print(TXT_LINE)


def remove_noise(signal_data):
    # Butterworth filter
    n = 1  # Filter order
    wn = 0.15  # Cutoff frequency
    btype, analog = sg.butter(n, wn, output='ba')

    # Applying the filter
    signal_reduced_noise = sg.filtfilt(btype, analog, signal_data)

    return signal_reduced_noise


def calculate_short_time_energy(signal_data):

    signal = np.array(signal_data, dtype=float)
    win = sg.get_window("hamming", 301)

    if isinstance(win, str):
        win = sg.get_window(win, max(1, len(signal) // 8))
    win = win / len(win)

    signal_short_time_energy = sg.convolve(signal ** 2, win ** 2, mode="same")

    return signal_short_time_energy
