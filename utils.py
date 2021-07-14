import librosa.display
import scipy
import soundfile as sf
import scipy.signal as sg
import noisereduce as nr

from constants import *
from plots import *


def pre_processing(signal_data, file_name):
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
    filtered_file_path = ".\\data\\samples\\" + file_name + "_filtered.wav"
    sf.write(filtered_file_path, signal_trimmed, DEFAULT_SAMPLE_RATE)

    # Print statistics
    print(TXT_LINE, "\n")
    print(TXT_STATISTICS)
    print(TXT_ORIGINAL_AUDIO_SAMPLE_RATE.format(DEFAULT_SAMPLE_RATE))
    print(TXT_ORIGINAL_AUDIO_DURATION_FORMAT.format(librosa.get_duration(signal_data)))
    print(TXT_TRIMMED_AUDIO_DURATION_FORMAT.format(librosa.get_duration(signal_trimmed)))
    print(TXT_ZCR_AVERAGE.format(zcr_average), "\n")
    print(TXT_LINE)

    return signal_trimmed


def remove_noise(signal_data):
    # Butterworth filter
    # n = 1  # Filter order
    # wn = 0.15  # Cutoff frequency
    # btype, analog = sg.butter(n, wn, output='ba')
    #
    # # Applying the filter
    # signal_reduced_noise = sg.filtfilt(btype, analog, signal_data)

    # perform noise reduction
    reduced_noise = nr.reduce_noise(audio_clip=signal_data,
                                    noise_clip=signal_data,
                                    verbose=False)

    return reduced_noise


def calculate_short_time_energy(signal_data):
    signal = np.array(signal_data, dtype=float)
    win = sg.get_window("hamming", 301)

    if isinstance(win, str):
        win = sg.get_window(win, max(1, len(signal) // 8))
    win = win / len(win)

    signal_short_time_energy = sg.convolve(signal ** 2, win ** 2, mode="same")

    return signal_short_time_energy


def digits_segmentation(signals):
    y = signals
    # duration = librosa.get_duration(y, sr=DEFAULT_SAMPLE_RATE)  # διαρκεια ήχουν
    signal_reverse = signals[::-1]  # αντιστροφη σηματος
    frame_duration_length = 0.03  # το μήκος ενός πλαισίου πλαισίου στο σήμα
    frame_step = 0.01  # το ολισθαίνον βήμα μέσα σε αυτό το πλαίσιο για 10ms.

    frame_length = round(frame_duration_length * DEFAULT_SAMPLE_RATE)  # frame length
    frame_step = round(frame_step * DEFAULT_SAMPLE_RATE)  # sliding step

    frames = librosa.onset.onset_detect(y, sr=DEFAULT_SAMPLE_RATE, hop_length=frame_length, backtrack=True)
    times = librosa.frames_to_time(frames, sr=DEFAULT_SAMPLE_RATE, hop_length=frame_length)
    samples = librosa.frames_to_samples(frames, frame_length)

    frames_reverse = librosa.onset.onset_detect(signal_reverse, sr=DEFAULT_SAMPLE_RATE, hop_length=frame_length,
                                                backtrack=True)
    times_reverse = librosa.frames_to_time(frames_reverse, sr=DEFAULT_SAMPLE_RATE, hop_length=frame_length)

    i = 0
    while i < len(times_reverse) - 1:
        times_reverse[i] = frame_duration_length - times_reverse[i]
        i += 1

    times_reverse = sorted(times_reverse)

    i = 0
    while i < len(times_reverse) - 1:
        if times_reverse[i + 1] - times_reverse[i] < 1:
            times_reverse = np.delete(times_reverse, i)
            i -= 1
        i += 1

    i = 0
    while i < len(times) - 1:
        if times[i + 1] - times[i] < 1:
            times = np.delete(times, i + 1)
            frames = np.delete(frames, i + 1)
            samples = np.delete(samples, i + 1)
            i = i - 1
        i = i + 1

    merged_onset_times = [*times, *times_reverse]
    merged_onset_times = sorted(merged_onset_times)

    samples = librosa.time_to_samples(merged_onset_times, sr=DEFAULT_SAMPLE_RATE)

    # spectrograph with detected onset spots todo μεταφορα κωδικα στο Plots
    plt.figure(5)
    plt.title('Spectroscopy with points resulting from onset')
    Y = librosa.stft(signals)
    Yto_db = librosa.amplitude_to_db(abs(Y))
    librosa.display.specshow(Yto_db, sr=DEFAULT_SAMPLE_RATE, x_axis='time', y_axis='hz')
    plt.vlines(merged_onset_times, 0, 10000, color='k')
    plt.show()
    return samples


def digit_recognition(signals, samples):
    i = 0
    # number of valid digits from onset detection
    count_digits = 0
    digit = {}
    while i < len(samples):
        if i == len(samples) - 1 and len(samples) % 2 == 1:
            digit[count_digits] = signals[samples[i - 1]:samples[
                i]]  # ipd.Audio(y[onset_samples[i]:onset_samples[i+1]],rate = s)
        else:
            digit[count_digits] = signals[samples[i]:samples[
                i + 1]]  # ipd.Audio(y[onset_samples[i]:onset_samples[i+1]],rate = s)
        count_digits += 1
        i += 2
    # song[numbSongs] = y[onset_samples[-1]:]#ipd.Audio(y[onset_samples[-1]:],rate = s)

    print('Total digits: ', len(digit))

    return digit
