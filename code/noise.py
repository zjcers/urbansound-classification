import os
import sys
import numpy
import random
import librosa
import common
import params

SAMPLE_RATE = 48000

def make_white():
    SAMPLES = 48000 * 1
    output = numpy.random.random(SAMPLES)
    return output

def load_noise_data(n_points):
    def name_of(i):
        return "white-noise-%s-%sms-%sbins.npy" % (i, params.MFCC_WINDOW_LENGTH, params.MFCC_BINS)
    try:
        os.mkdir("tmp")
    except FileExistsError:
        pass
    for i in range(n_points):
        sys.stdout.write("Loading noise sample %s of %s: " % (i, n_points))
        sys.stdout.flush()
        try:
            mfcc = numpy.load(os.path.join("tmp", name_of(i)))
            sys.stdout.write("cached!\r")
            sys.stdout.flush()
            for slice in common.mfcc_reshape(mfcc):
                yield "white_noise", slice
        except FileNotFoundError:
            noise = make_white()
            mfcc = common.process_mfcc(noise, SAMPLE_RATE)
            numpy.save(os.path.join("tmp", name_of(i)), mfcc)
            sys.stdout.write("done!\r")
            sys.stdout.flush()
            for slice in common.mfcc_reshape(mfcc):
                yield "white_noise", slice
    print()
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    noise = make_white()
    bins = 16
    fft = numpy.fft.fft(noise, 16)
    plt.plot(range(bins), fft)
    plt.show()