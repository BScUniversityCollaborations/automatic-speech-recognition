import librosa.display
import noisereduce as nr
import soundfile as sf
import scipy.signal as sg

from plots import *


def pre_processing(signal_data, file_name):
    # === Pre-Emphasis ===
    signal_emphasized = librosa.effects.preemphasis(signal_data)

    # === Filtering ===
    # Remove the background noise from the audio file.
    signal_reduced_noise = remove_noise(signal_data)

    # Remove the silent parts of the audio that are less than 40dB
    signal_filtered, i = librosa.effects.trim(signal_reduced_noise, TOP_DB)

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

    frame_length = round(WINDOW_LENGTH * DEFAULT_SAMPLE_RATE)

    frames = librosa.onset.onset_detect(signal_nparray, sr=DEFAULT_SAMPLE_RATE, hop_length=frame_length, backtrack=True)
    times = librosa.frames_to_time(frames, sr=DEFAULT_SAMPLE_RATE, hop_length=frame_length)
    samples = librosa.frames_to_samples(frames, frame_length)

    frames_reverse = librosa.onset.onset_detect(signal_reverse, sr=DEFAULT_SAMPLE_RATE, hop_length=frame_length,
                                                backtrack=True)
    times_reverse = librosa.frames_to_time(frames_reverse, sr=DEFAULT_SAMPLE_RATE, hop_length=frame_length)

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
    # number of valid digits from onset detection TODO change the comment
    count_digits = 0
    digit = {}
    while i < len(samples):
        if i == len(samples) - 1 and len(samples) % 2 == 1:
            digit[count_digits] = signal_data[samples[i - 1]:samples[i]]
        else:
            digit[count_digits] = signal_data[samples[i]:samples[i + 1]]
        count_digits += 1
        i += 2

    return digit
