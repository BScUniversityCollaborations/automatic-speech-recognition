# Made by P17172, P17168, P17164

import os
import sys
import librosa
import soundfile as sf

from constants import *
from utils import *

# Get the sound file path from the user
filePath = input(TXT_INPUT_FILE)
# filePath = "G:\\Downloads (HDD)\\University\\8th Semester\\SPEECH AND AUDIO PROCESSING\\ΕΡΓΑΣΙΕΣ\\[aalykiot] " \
#           "automatic-speech-recognition\\samples\\tests\\test-1.m4a"

# Check if file exists
if os.path.exists(filePath) is False:
    sys.exit(TXT_FILE_NOT_FOUND + str(filePath))

# Check if file is mp3 or wav
if filePath.endswith(".mp3") or filePath.endswith(".wav") or filePath.endswith(".m4a") is not True:
    sys.exit(TXT_FILE_WRONG_EXTENSION + str(filePath))

# ==== ANALYSIS PROCESS ====
# Load the file from path, then get the signal and sample rate.
signal, sample_rate = librosa.load(filePath)

# Remove the background noise from the audio file.
signal_reduced_noise = remove_noise(signal)

# Remove the silent parts of the audio that are less than 40dB
signal_trimmed, i = librosa.effects.trim(signal_reduced_noise, TOP_DB)

# Print statistics
print(TXT_LINE, "\n")
print(TXT_STATISTICS)
print(TXT_ORIGINAL_AUDIO_SAMPLE_RATE.format(sample_rate))
print(TXT_ORIGINAL_AUDIO_DURATION_FORMAT.format(librosa.get_duration(signal)))
print(TXT_TRIMMED_AUDIO_DURATION_FORMAT.format(librosa.get_duration(signal_trimmed)), "\n")
print(TXT_LINE)

# sf.write("test.wav", signal_trimmed, sample_rate)
