# Made by P17172, P17168, P17164

import os
import sys
import librosa
import soundfile as sf

from constants import *
from utils import *

# Get the sound file path from the user
file_path = input(TXT_INPUT_FILE)

# Check if file exists
if os.path.exists(file_path) is False:
    sys.exit(TXT_FILE_NOT_FOUND + str(file_path))

# Check if file is mp3 or wav
root, extension = os.path.splitext(file_path)
if extension not in AUDIO_FILE_EXTENSIONS:
    sys.exit(TXT_FILE_WRONG_EXTENSION + str(file_path))

# ==== ANALYSIS PROCESS ====
# Load the file from path, then get the signal and sample rate.
signal, sample_rate = librosa.load(file_path)

# Remove the background noise from the audio file.
signal_reduced_noise = remove_noise(file_path, signal, sample_rate)

# Remove the silent parts of the audio that are less than 40dB
signal_trimmed, i = librosa.effects.trim(signal_reduced_noise, TOP_DB)

# Print statistics
print(TXT_LINE, "\n")
print(TXT_STATISTICS)
print(TXT_ORIGINAL_AUDIO_SAMPLE_RATE.format(sample_rate))
print(TXT_ORIGINAL_AUDIO_DURATION_FORMAT.format(librosa.get_duration(signal)))
print(TXT_TRIMMED_AUDIO_DURATION_FORMAT.format(librosa.get_duration(signal_trimmed)), "\n")
print(TXT_LINE)


# todo remove this
# sf.write("test.wav", signal_trimmed, sample_rate)
