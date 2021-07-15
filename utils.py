import math

import librosa.display
import noisereduce as nr
import soundfile as sf
import scipy.signal as sg
from sklearn.model_selection import train_test_split

from plots import *


def pre_processing(signal_data, file_name):
    # === Pre-Emphasis ===
    signal_emphasized = librosa.effects.preemphasis(signal_data)

    # === Filtering ===
    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, TOP_DB)

    signal_zcr = librosa.feature.zero_crossing_rate(signal_filtered)
    zcr_average = np.mean(signal_zcr)

    signal_short_time_energy = calculate_short_time_energy(signal_filtered)

    # Show plots
    show_plot_emphasized(signal_data, signal_emphasized)
    show_plots_compare_two_signals(signal_data, signal_reduced_noise)
    show_plot_zcr(signal_zcr)
    show_plot_short_time_energy(signal_filtered, signal_short_time_energy)

    # Exporting the filtered audio file.
    filtered_file_path = ".\\data\\samples\\" + file_name + "_filtered.wav"
    sf.write(filtered_file_path, signal_filtered, DEFAULT_SAMPLE_RATE)

    # Print statistics
    print(TXT_LINE, "\n")
    print(TXT_PRE_PROCESSING_STATISTICS)
    print(TXT_ORIGINAL_AUDIO_SAMPLE_RATE.format(DEFAULT_SAMPLE_RATE))
    print(TXT_AUDIO_ORIGINAL_DURATION_FORMAT.format(
        round(librosa.get_duration(signal_data, sr=DEFAULT_SAMPLE_RATE), 2))
    )
    print(TXT_AUDIO_FILTERED_DURATION_FORMAT.format(
        round(librosa.get_duration(signal_filtered, sr=DEFAULT_SAMPLE_RATE), 2))
    )
    print(TXT_ZCR_AVERAGE.format(zcr_average), "\n")
    print(TXT_LINE)

    return signal_filtered


def remove_noise(signal_data):
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


def digits_segmentation(signal_nparray):
    # We reverse the signal nparray.
    signal_reverse = signal_nparray[::-1]

    frames = librosa.onset.onset_detect(signal_nparray, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH, backtrack=True)
    times = librosa.frames_to_time(frames, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH)
    samples = librosa.frames_to_samples(frames, FRAME_LENGTH)

    frames_reverse = librosa.onset.onset_detect(signal_reverse, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH,
                                                backtrack=True)
    times_reverse = librosa.frames_to_time(frames_reverse, sr=DEFAULT_SAMPLE_RATE, hop_length=FRAME_LENGTH)

    i = 0
    while i < len(times_reverse) - 1:
        times_reverse[i] = WINDOW_LENGTH - times_reverse[i]
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

    merged_times = [*times, *times_reverse]
    merged_times = sorted(merged_times)

    samples = librosa.time_to_samples(merged_times, sr=DEFAULT_SAMPLE_RATE)

    return samples


def digit_recognition(signal_data, samples):
    i = 0
    # number of valid digits from onset detection todo
    count_digits = 0
    digit = {}

    # range(start_from, stop_at, step_size)
    for i in range(0, len(samples), 2):
        print(i)
        if len(samples) % 2 == 1 and i == len(samples) - 1:
            digit[count_digits] = signal_data[samples[i - 1]:samples[i]]
        else:
            digit[count_digits] = signal_data[samples[i]:samples[i + 1]]
        count_digits += 1

    return digit


def recognition(digits, signal_data, dataset):
    # Init an array that will contain our recognized digits in string.
    recognized_digits_array = []
    for digit in digits:
        mfcc_input = librosa.feature.mfcc(digit, signal_data, hop_length=FRAME_LENGTH, n_mfcc=13)
        mfcc_input_mag = librosa.amplitude_to_db(abs(mfcc_input))

        Dnew = []
        mfccs = []
        # 0-9 from training set
        for i in range(len(dataset)):
            # We basically filter the training dataset as well.
            dataset[i] = filter_signal(dataset[i])
            # MFCC for each digit from the training set
            mfcc = librosa.feature.mfcc(dataset[i], signal_data, hop_length=80, n_mfcc=13)
            # logarithm of the features ADDED
            mfcc_mag = librosa.amplitude_to_db(abs(mfcc))
            # apply dtw
            D, wp = librosa.sequence.dtw(X=mfcc_input_mag, Y=mfcc_mag, metric='euclidean', backtrack=True)
            # make a list with minimum cost of each digit
            Dnew.append(D[-1, -1])
            mfccs.append(mfcc_mag)

        # index of MINIMUM COST
        index = Dnew.index(min(Dnew))

        show_mel_spectrogram()

    return recognized_digits_array


def get_training_samples_signal():
    # Initialize an array to append the signals of the training samples.
    training_samples_signals = []

    # Loop between a range of 0-9, 0 in range(10) is 0 to 9 in python.
    for i in range(10):
        # Loop between the labels, s1 means sample1 and so on.
        for name in DATASET_SPLIT_LABELS:
            # Add the signal to our array.
            training_samples_signals.append(librosa.load("\\data\\training\\"
                                                         + str(i + 1)
                                                         + "_"
                                                         + name
                                                         + AUDIO_WAV_EXTENSION,
                                                         sr=DEFAULT_SAMPLE_RATE))

    return training_samples_signals


def filter_signal(signal_data):
    # === Filtering ===
    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, TOP_DB)
    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, _ = librosa.effects.trim(signal_reduced_noise, TOP_DB)

    return signal_filtered


def cross_validation(labels, mfcc):
    training, testing = train_test_split(labels, test_size=0.3, shuffle=True)

    D = np.ones((len(labels), len(labels))) * -1

    score = 0.0
    for i in range(len(testing)):
        x = mfcc[i]
        dmin, jmin = math.inf, -1

        for j in range(len(training)):
            y = mfccs[j]
            d = D[i, j]

            if d.all() == -1:
                d = librosa.sequence.dtw(x, y)

            if d.all() < dmin:
                dmin = d
                jmin = j

        score += 1.0 if (labels[i] == labels[jmin]) else 0.0

    print('Rec rate {}%'.format(100. * score / len(testing)))
