# Made by P17172, P17168, P17164

import os
import sys

import numpy

from utils import *

# Get the sound file path from the user
file_path = input(TXT_INPUT_FILE)

# Check if file exists
if os.path.exists(file_path) is False:
    sys.exit(TXT_FILE_NOT_FOUND + str(file_path))

# Check if file is mp3 or wav
root, extension = os.path.splitext(file_path)
if extension not in AUDIO_WAV_EXTENSION:
    sys.exit(TXT_FILE_WRONG_EXTENSION + str(file_path))

# Load the file from path, then get the signal and sample rate.
signal, sr = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE)

# === Start Pre-Processing ===
pre_processing(signal)

# todo remove this
# sf.write("test.wav", signal_trimmed, sample_rate)
