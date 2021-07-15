# Made by P17172, P17168, P17164

import os

from utils import *

root = ""
while True:
    # Get the sound file path from the user
    file_path = input(TXT_INPUT_FILE)

    # Check if file exists
    if os.path.exists(file_path) is False:
        print(TXT_FILE_NOT_FOUND + str(file_path))
        print()
        continue  # Return to the start of the loop

    # Check if file is mp3 or wav
    root, extension = os.path.splitext(file_path)
    if extension not in AUDIO_WAV_EXTENSION:
        print(TXT_FILE_WRONG_EXTENSION + str(file_path))
        print()
        continue  # Return to the start of the loop

    break

# if plots directory doesn't exists, create is so we save our plots.
if os.path.exists(DIRECTORY_PLOTS) is False:
    os.mkdir(DIRECTORY_PLOTS)

# Load the file from path, then get the signal and sample rate.
signal, sr = librosa.load(file_path, sr=DEFAULT_SAMPLE_RATE)

# === Start Pre-Processing ===
pre_proceed_signal = pre_processing(signal, os.path.basename(root))

# === Start digit segmentation ===
samples = digits_segmentation(pre_proceed_signal)

# === Feature extraction & word recognition ===
digits_array = digit_recognition(pre_proceed_signal, samples)



# === Display words a list of words found ===
# Prints the list that contains all the words found and separates each word
# with a ", " excluding the last one.
print(TXT_DIGITS_FOUND.format(len(digits_array)))
print("\n", TXT_DIGITS_RECOGNIZED)
print(", ".join([str(i) for i in digits_array]))
