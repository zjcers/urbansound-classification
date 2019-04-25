import csv
import itertools
import os
import sys
import concurrent.futures as concur
import numpy
import soundfile
import librosa
import librosa.feature
import params

def ms_to_samples(sample_rate, ms):
    return (ms*sample_rate) // 1000

def process_mfcc(y, sr, window_length = params.MFCC_WINDOW_LENGTH, n_mfcc = params.MFCC_BINS):
    hop_length = ms_to_samples(sr, window_length)
    mfcc = librosa.feature.mfcc(y=y,
                                sr=sr,
                                hop_length=hop_length,
                                n_mfcc=n_mfcc)
    return mfcc

def mfcc_reshape(mfcc: numpy.ndarray, depth=params.MFCC_CNN_DEPTH):
    """
    Reshapes MFCCs to be of fixed size
    """
    for lower in range(0, mfcc.shape[1], depth // 8):
        upper = lower + depth
        if upper < mfcc.shape[1]:
            # print("Slicing from", lower, "to", upper, "from", mfcc.shape)
            yield mfcc[::, lower:upper]

def load_wav_mono(filename):
    y, sr = soundfile.read(filename)
    if len(y.shape) == 2:
        y.reshape(2, y.shape[0])
        y = librosa.core.to_mono(y)
    return y, sr

def find_mfcc(filename, data_dir="tmp"):
    try:
        os.makedirs(os.path.join(data_dir, os.path.dirname(filename)))
    except FileExistsError:
        pass
    sys.stdout.write("Getting MFCC for " + filename + ": ")
    sys.stdout.flush()
    full_name = os.path.join(data_dir, filename + "-" + str(params.MFCC_WINDOW_LENGTH) + "ms-"+str(params.MFCC_BINS)+"bins.npy")
    try:
        mfcc = numpy.load(full_name, mmap_mode='r')
        sys.stdout.write("loaded from cache!\r")
        sys.stdout.flush()
        return mfcc
    except FileNotFoundError:
        sys.stdout.write("processing")
        sys.stdout.flush()
        y, sr = load_wav_mono(filename)
        mfcc = process_mfcc(y, sr)
        sys.stdout.write(", saving to cache\r")
        sys.stdout.flush()
        numpy.save(full_name, mfcc)
        return mfcc

SLICE_FILE_NAME = 0
FOLD = 5
CLASS_ID = 6

def augment_row(row):
    original_path = "UrbanSound8K/audio/fold%s/" % (row[FOLD],)
    try:
        os.makedirs(os.path.join("tmp", original_path))
    except FileExistsError:
        pass
    augments = []
    NOISE_SCALE = 0.001
    SHIFT_SCALE = 0.1
    both_channels, sr = soundfile.read(os.path.join(original_path, row[SLICE_FILE_NAME]))
    chans = range(1)
    if len(both_channels.shape) == 2:
        chans = range(both_channels.shape[1])
        both_channels = both_channels.reshape(both_channels.shape[1], both_channels.shape[0])
    else:
        both_channels = both_channels.reshape(1, both_channels.shape[0])
    for chan in chans:
        original_data = both_channels[chan]
        chan_lbl = "left" if chan == 0 else "right"
        for shift in range(0, 1):
            shift_scalar = SHIFT_SCALE * shift + 1.0
            for noise_level in range(0, 3, 2):
                noise_scalar = NOISE_SCALE * noise_level
                for pat in range(1):
                    name = os.path.join("tmp", original_path, "%s-shift-%s-noise-lvl%s-pat%s-%s" % (chan_lbl, shift_scalar, noise_scalar, pat, row[SLICE_FILE_NAME]))
                    if not os.path.exists(name):
                        noisy_data = original_data
                        if noise_level != 0:
                            noise = numpy.random.randn(len(original_data))
                            noisy_data = original_data + noise_scalar * noise
                        shifted_data = noisy_data
                        if shift != 0:
                            shifted_data = librosa.effects.pitch_shift(noisy_data, sr, shift_scalar)
                        assert len(shifted_data) >= len(both_channels)
                        soundfile.write(name, shifted_data, sr)
                    yield name
def process_file(fn):
    mfcc = find_mfcc(fn)
    for slice in mfcc_reshape(mfcc):
        yield slice

def process_row(row):
    slices = itertools.chain.from_iterable(map(process_file, augment_row(row)))
    return zip(itertools.repeat(row[CLASS_ID]), slices)

def load_data(label_file_name="UrbanSound8K/metadata/UrbanSound8K.csv"):
    with open(label_file_name, 'r') as labels:
        reader = csv.reader(labels)
        next(reader)
        augmented_rows = itertools.chain.from_iterable(map(process_row, reader))
        print()
        cats = []
        samples = []
        for row in augmented_rows:
            cats.append(row[0])
            samples.append(row[1])
        return cats, samples

def print_header(text):
    print(text.center(80, ' '))
    print('=' * 80)

def unique(iterable):
    """
    Filters iterable down to unique items while preserving order
    """
    seen = set()
    for item in iterable:
        if item not in seen:
            seen.add(item)
            yield item
