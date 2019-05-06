import csv
import itertools
import os
import sys
import math
import numpy
import soundfile
import librosa
import librosa.feature
from keras.utils import to_categorical
import params


def ms_to_samples(sample_rate, ms):
    return (ms*sample_rate) // 1000


def process_mfcc(y, sr):
    window_length = ms_to_samples(sr, params.MFCC_WINDOW_LENGTH)
    hop_length = ms_to_samples(sr, params.MFCC_HOP_LENGTH)
    S = librosa.core.stft(y,
                          n_fft=window_length,
                          hop_length=hop_length,
                          center=False)
    mfcc = librosa.feature.mfcc(S=S,
                                fmin=params.MFCC_FMIN,
                                hop_length=hop_length,
                                n_mfcc=params.MFCC_BINS,
                                n_fft=window_length)
    return numpy.real(mfcc)


def compute_logE(y, sr):
    window_length = ms_to_samples(sr, params.MFCC_WINDOW_LENGTH)
    hop_length = ms_to_samples(sr, params.MFCC_HOP_LENGTH)
    frames = librosa.util.frame(y,
                                frame_length=window_length,
                                hop_length=hop_length)
    output = numpy.zeros(frames.shape[1])
    # From ETSI ES 201 108
    for frame in range(frames.shape[1]):
        try:
            squared = numpy.real(frames[:, frame])**2
            summed = numpy.sum(squared)
            power = math.log(summed)
            output[frame] = power
        except ValueError:
            # silence the math domain errors
            output[frame] = 0.0
    return output


def process_features(y, sr):
    y, sr = to_mono(y, sr)
    mfcc = process_mfcc(y, sr)
    mfcc = mfcc.reshape((mfcc.shape[1], mfcc.shape[0]))
    logE = compute_logE(y, sr)
    logE = numpy.reshape(logE, (logE.shape[0], 1))
    return numpy.concatenate((mfcc, logE), axis=1)

def slice_inputs(mfcc: numpy.ndarray, depth=1):
    """
    Reshapes MFCCs to be of fixed size
    """
    for lower in range(0, mfcc.shape[0], depth):
        upper = lower + depth
        if upper < mfcc.shape[0]:
            yield mfcc[lower:upper, :]


def to_mono(y, sr):
    if len(y.shape) == 2:
        y = y.reshape(2, y.shape[0])
        y = librosa.core.to_mono(y)
    return y, sr

SLICE_FILE_NAME = 0
FOLD = 5
CLASS_ID = 6
