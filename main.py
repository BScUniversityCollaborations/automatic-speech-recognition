# Made by P17172, P17168, P17164

import os
import sys

from constants import *

# Get the sound file path from the user
filePath = input(INPUT_FILE)
# Check if file exists
if os.path.exists(filePath) is False:
    sys.exit(FILE_NOT_FOUND + str(filePath))



