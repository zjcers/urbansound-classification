import csv
import itertools
import os
import sys
import concurrent.futures as concur
import numpy
import soundfile
import librosa
import librosa.feature
from keras.utils import to_categorical
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
    for lower in range(0, mfcc.shape[1], depth // params.MFCC_OVERLAP_FACTOR):
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

def find_mfcc(filename, y, sr, data_dir="tmp"):
    try:
        os.makedirs(os.path.join(data_dir, os.path.dirname(filename)))
    except FileExistsError:
        pass
    sys.stdout.write("Getting MFCC for " + filename + "\r")
    sys.stdout.flush()
    full_name = os.path.join(data_dir, filename + "-" + str(params.MFCC_WINDOW_LENGTH) + "ms-"+str(params.MFCC_BINS)+"bins.npy")
    try:
        mfcc = numpy.load(full_name, mmap_mode='r')
        return mfcc
    except FileNotFoundError:
        # y, sr = load_wav_mono(filename)
        mfcc = process_mfcc(y, sr)
        numpy.save(full_name, mfcc)
        return numpy.load(full_name, mmap_mode='r') # don't keep the other mfcc around to keep memory usage in check
        # return mfcc

SLICE_FILE_NAME = 0
FOLD = 5
CLASS_ID = 6

def augment(original_fn):
    # try:
    #     os.makedirs(os.path.join("tmp", os.path.dirname(original_path)))
    # except FileExistsError:
    #     pass
    NOISE_SCALE = 0.001
    SHIFT_SCALE = 0.1
    both_channels, sr = soundfile.read(original_fn)
    chans = range(1)
    if len(both_channels.shape) == 2:
        chans = range(both_channels.shape[1])
        both_channels = both_channels.reshape(both_channels.shape[1], both_channels.shape[0])
    else:
        both_channels = both_channels.reshape(1, both_channels.shape[0])
    for chan in chans:
        original_data = both_channels[chan]
        chan_lbl = "left" if chan == 0 else "right"
        for shift in range(params.PITCH_LOWER, params.PITCH_UPPER + 1):
            shift_scalar = SHIFT_SCALE * shift + 1.0
            for noise_level in range(0, params.NOISE_LEVELS + 1):
                noise_scalar = NOISE_SCALE * noise_level
                for pat in range(params.NOISE_PATTERNS + 1):
                    name = os.path.join("tmp", "%s-shift-%s-noise-lvl%s-pat%s-%s" % (chan_lbl, shift_scalar, noise_scalar, pat, original_fn))
                    # if not os.path.exists(name):
                    noisy_data = original_data
                    if noise_level != 0:
                        noise = numpy.random.randn(len(original_data))
                        noisy_data = original_data + noise_scalar * noise
                    shifted_data = noisy_data
                    if shift != 0:
                        shifted_data = librosa.effects.pitch_shift(noisy_data, sr, shift_scalar)
                    assert len(shifted_data) >= len(both_channels)
                    # soundfile.write(name, shifted_data, sr)
                    yield sr, shifted_data, name
def process_file(fn, y, sr):
    mfcc = find_mfcc(fn, y, sr)
    for slice in mfcc_reshape(mfcc):
        yield slice

def process_row(row):
    path = os.path.join("UrbanSound8K", "audio", "fold%s" % (row[FOLD],), row[SLICE_FILE_NAME])
    slices = []
    for sr, y, name in augment(path):
        slices.extend(process_file(name, y, sr))
    return zip(itertools.repeat(row[CLASS_ID]), slices)

def load_data(label_file_name="UrbanSound8K/metadata/UrbanSound8K.csv", fold_selector=lambda _: True):
    with open(label_file_name, 'r') as labels:
        reader = csv.reader(labels)
        next(reader)
        just_this_fold = filter(fold_selector, reader)
        augmented_rows = itertools.chain.from_iterable(map(process_row, just_this_fold))
        print()
        cats = []
        samples = []
        for row in augmented_rows:
            cats.append(row[0])
            samples.append(row[1])
        return samples, to_categorical(cats, 10)

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

def add_dim(arr: numpy.ndarray):
    old_shape = arr.shape
    return arr.reshape(old_shape[0], old_shape[1], 1)

def get_data(test_fold=1):
    train_folds = [x for x in range(1, 3) if x != test_fold]
    x_train = []
    y_train = []
    test_fold = str(test_fold)
    x_test, y_test = load_data(fold_selector=lambda row: row[FOLD] == test_fold)
    for fold in train_folds:
        x_fold, y_fold = load_data(fold_selector=lambda row: row[FOLD] != test_fold)
        x_train.extend(x_fold)
        y_train.extend(y_fold)
    x_train = numpy.stack(list(map(add_dim, x_train)), axis=0)
    y_train = numpy.stack(y_train)
    x_test = numpy.stack(list(map(add_dim, x_test)), axis=0)
    numpy.save("tmp/x_train-%s.npy" % (test_fold,), x_train)
    numpy.save("tmp/y_train-%s.npy" % (test_fold,), y_train)
    numpy.save("tmp/x_test-%s.npy" % (test_fold,), x_test)
    numpy.save("tmp/y_test-%s.npy" % (test_fold,), y_test)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = numpy.load("tmp/x_train-%s.npy" % (test_fold,), mmap_mode='r')
    y_train = numpy.load("tmp/y_train-%s.npy" % (test_fold,), mmap_mode='r')
    x_test = numpy.load("tmp/x_test-%s.npy" % (test_fold,), mmap_mode='r')
    y_test = numpy.load("tmp/y_test-%s.npy" % (test_fold,), mmap_mode='r')
    return x_train, y_train, x_test, y_test
