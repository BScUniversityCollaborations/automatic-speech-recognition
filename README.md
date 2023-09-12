# Automatic Speech Recognition (ASR)
Created an ASR (Automatic Speech Recognition) system that takes in individual recordings. Each recording represents a sentence composed of 5-10 English language digits, separated by adequate pauses. The system involves segmenting the sentence using a classifier, differentiating between background and foreground sounds. Finally, the system identifies each word exclusively based on the mel-spectrogram as its spectral representation.

**Exercise:**
You are asked to implement an ASR system, which accepts input one recording at a time, which constitutes a sentence consisting of 5-10 digits of the English language spoken with sufficiently long pauses.
The system proceeds to segment the sentence using a mandatory background vs foreground classifier of your choice.
It then recognizes each word using only the mel-spectrogram as its spectral representation. If you need training data, use only dataset(s) from the OpenSLR site.
At the output, a text is produced with the recognized digits. 
• Emphasize signal processing, before the segmentation/identification steps begin (eg, with appropriate filters, changing the sample rate, etc.). 
• It is important to describe the system algorithmically (feature extraction, recognition algorithm) and explain its performance using appropriate metrics. 
• You must explain what data you used when testing and training the system. If yours, how did you create them?
• Try not to make the system dependent on the characteristics of the speaker's voice, but to be as speaker-independent as possible
