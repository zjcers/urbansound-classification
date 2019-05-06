import itertools
import random
import os
import soundfile
import csv
import sys
from keras.utils import to_categorical
import keras
import numpy
import common
import keras.callbacks
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import LSTM
import params


class DataGenerator(keras.utils.Sequence):
    """
    """
    class Rows:
        SLICE_FILE_NAME = 0
        FOLD = 5
        CLASS_ID = 6
    def __init__(self, batch_size, test_fold, validation=False, path="UrbanSound8K", depth=1):
        self.batch_size = batch_size
        self.test_fold = test_fold
        self.validation = validation
        self.path = path
        self._depth = depth
        self.wavs = list(self._list_wavs())
        self.outputshape = (params.BATCH_SIZE, params.MFCC_BINS+1, self._depth)
        if depth > 1: # CNN
            self.outputshape = self.outputshape + (1,)
    def _list_wavs(self):
        """
        Reads the dataset CSV file and yields tuples of (path, fold, classid)
        """
        with open(os.path.join(self.path, "metadata", "UrbanSound8K.csv"), 'r') as metadata:
            reader = csv.reader(metadata)
            next(reader) # skip header
            for row in reader:
                fold = row[self.Rows.FOLD]
                folds_eq = fold == self.test_fold
                if (self.validation and folds_eq) or (not self.validation and not folds_eq):
                    path = os.path.join(self.path, "audio", "fold%s" % (fold,), row[self.Rows.SLICE_FILE_NAME])
                    yield (path, fold, int(row[self.Rows.CLASS_ID]))
    def __len__(self):
        slices = 0
        for (path, _, _) in self.wavs:
            with soundfile.SoundFile(path) as snd:
                frame_length = common.ms_to_samples(snd.samplerate, params.MFCC_WINDOW_LENGTH)
                hop_length = common.ms_to_samples(snd.samplerate, params.MFCC_HOP_LENGTH)
                # From librosa.util.frame
                n_frames = 1 + int((len(snd) - frame_length) / hop_length)
                slices += n_frames
        return slices // self.batch_size
    def hash_int(self, n):
        return (n * 2654435761) & 0xFFFFFFFF # 2^32 / Golden Ratio
    def slice(self, wavIdx, thickness):
        path, fold, class_ = self.wavs[wavIdx]
        y, sr = soundfile.read(path)
        inputs = common.process_features(y, sr)
        c = 0
        for slice_ in common.slice_inputs(inputs, thickness):
            c += 1
            yield slice_, class_
    def __getitem__(self, idx):
        inputs = numpy.empty((params.BATCH_SIZE, self._depth, params.MFCC_BINS+1))
        outputs = []
        idx = self.hash_int(idx) % len(self.wavs)
        outputIdx = 0
        sourceIdxs = itertools.chain(range(idx, len(self.wavs)), \
                     itertools.chain.from_iterable(
                         itertools.repeat(
                             range(0, len(self.wavs)))))
        while outputIdx < params.BATCH_SIZE:
            srcIdx = next(sourceIdxs)
            for inputSlice,outputSlice in self.slice(srcIdx, self._depth):
                inputs[outputIdx] = inputSlice
                outputs.append(outputSlice)
                outputIdx += 1
                if outputIdx >= params.BATCH_SIZE:
                    break
        return numpy.reshape(inputs, self.outputshape), \
               keras.utils.to_categorical(outputs, 10)
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(params.MFCC_BINS+1, params.MFCC_CNN_DEPTH, 1), activation="sigmoid"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(16, (3, 3), activation="sigmoid"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(optimizer='adadelta',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def lstm_model():
    model = Sequential()
    model.add(LSTM(params.MFCC_CNN_DEPTH, input_shape=(params.MFCC_BINS + 1, 1)))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(optimizer='adadelta',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def train_model(model, depth):
    checkpoints = keras.callbacks.ModelCheckpoint("model-%s-{epoch:02d}.hdf5" % (sys.argv[1],), save_best_only=True, period=2)
    earlystop = keras.callbacks.EarlyStopping(monitor="val_acc", min_delta=0.01, patience=12, restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir=("./logs-%s" % (sys.argv[1],)))
    train_gen = DataGenerator(params.BATCH_SIZE, "10", depth=depth)
    test_gen = DataGenerator(params.BATCH_SIZE, "10", validation=True, depth=depth)
    model.fit_generator(generator=train_gen,
              validation_data=test_gen,
              steps_per_epoch=len(train_gen),
              use_multiprocessing=True,
              callbacks=[checkpoints, earlystop, tensorboard],
              epochs=200)
    return model
model = None
depth = params.MFCC_CNN_DEPTH
if sys.argv[1] == "cnn":
    model = cnn_model()
elif sys.argv[1] == "lstm":
    model = lstm_model()
    depth = 1
else:
    print("Usage: python train.py (cnn|lstm)")
    exit(1)
train_model(model, depth)
model.save("urbansound-%s.hdf5" % (sys.argv[1],))
