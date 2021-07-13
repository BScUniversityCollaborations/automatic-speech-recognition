import numpy as np
import pyACA


def algorithm_zcr(signal, block_length, hop_length, sample_rate):
    # create blocks
    # signal/block length in samples/hop length in samples
    signal_blocks = pyACA.ToolBlockAudio(signal, block_length, hop_length)

    # number of results
    num_of_blocks = signal_blocks.shape[0]

    # compute time stamps
    t = (np.arange(0, num_of_blocks) * hop_length + (block_length / 2)) / sample_rate

    # allocate memory
    vzc = np.zeros(num_of_blocks)

    for n, block in enumerate(signal_blocks):
        # calculate the zero crossing rate
        vzc[n] = 0.5 * np.mean(np.abs(np.diff(np.sign(block))))

    return vzc, t  # vzc zero crossing rate / t time stamp
