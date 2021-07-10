# Made by P17172, P17168, P17164

import os
import sys
import librosa

from constants import *

# Get the sound file path from the user
filePath = input(INPUT_FILE)
# Check if file exists
if os.path.exists(filePath) is False:
    sys.exit(FILE_NOT_FOUND + str(filePath))

# Check if file is mp3 or wav
if filePath.endswith(".mp3") or filePath.endswith(".wav") or filePath.endswith(".m4a") is not True:
    sys.exit(FILE_WRONG_EXTENSION + str(filePath))

signal, sr = librosa.load(filePath)
signal_trimmed, i = librosa.effects.trim(TOP_DB, signal)

print(signal + "\n" + sr + "\n" + signal_trimmed + "\n" + i)

