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


common.print_header("Loading Data")
# labels, data = common.load_data()

# print("Loaded", len(data),"items")

# common.print_header("Calculating category vectors")
# category_vecs = to_categorical(labels)

# # Squish all of the MFCCs together into one numpy array
# def add_dim(arr: numpy.ndarray):
#     old_shape = arr.shape
#     return arr.reshape(old_shape[0], old_shape[1], 1)
# xs = numpy.stack(list(map(add_dim, data)), axis=0)
# #xs = numpy.stack(data, axis=0)
# print("Input data shape:", xs.shape)

class DataGenerator(keras.utils.Sequence):
    """
    """
    class Rows:
        SLICE_FILE_NAME = 0
        FOLD = 5
        CLASS_ID = 6
    def __init__(self, batch_size, test_fold, validation=False, path="UrbanSound8K", depth=params.MFCC_CNN_DEPTH):
        self.batch_size = batch_size
        self.test_fold = test_fold
        self.validation = validation
        self.path = path
        self._depth = depth
        self._length = 0
        self._cache = []
    def _list_wavs(self, for_train = True):
        with open(os.path.join(self.path, "metadata", "UrbanSound8K.csv"), 'r') as metadata:
            reader = csv.reader(metadata)
            next(reader) # skip header
            for row in reader:
                folds_eq = row[self.Rows.FOLD] == self.test_fold
                if (for_train and not folds_eq) or (not for_train and folds_eq):
                    yield (row[self.Rows.SLICE_FILE_NAME], row[self.Rows.FOLD], int(row[self.Rows.CLASS_ID]))
    def _get_all(self):
        if self._length > 0:
            return self._length
        l = 0
        for _ in self._sample_gen():
            l += 1
        self._cache = [None for _ in range(l)]
        self._length = l
        return l
    def __len__(self):
        total_batches = self._get_all()
        print("Will generate", total_batches)
        return total_batches
    def _sample_gen_slicer(self):
        for wav in self._list_wavs():
            for sr, y, name in common.augment(self._path_to(wav)):
                mfcc = common.find_mfcc(name, y, sr)
                for slice in common.mfcc_reshape(mfcc, self._depth):
                    yield slice, wav[2]
    def mkdir(self, filename):
        try:
            os.makedirs(os.path.dirname(filename))
        except FileExistsError:
            pass
    def _sample_gen(self):
        inputs = []
        targets = []
        amount_in_batch = 0
        idx = 0
        for slice, class_ in self._sample_gen_slicer():
            path_mfcc, path_tags = self._path_to_batch(idx)
            amount_in_batch += 1
            if os.path.exists(path_mfcc):
                if amount_in_batch == params.BATCH_SIZE:
                    amount_in_batch = 0
                    idx += 1
                    yield
            else:
                inputs.append(slice)
                targets.append(class_)
                if amount_in_batch == params.BATCH_SIZE:
                    cooked_inputs = numpy.stack(list(map(common.add_dim, inputs)), axis=0)
                    cooked_targets = keras.utils.to_categorical(targets, 10)
                    self.mkdir(path_mfcc)
                    numpy.save(path_mfcc, cooked_inputs)
                    numpy.save(path_tags, cooked_targets)
                    yield
                    idx += 1
                    amount_in_batch = 0
                    inputs.clear()
                    targets.clear()
    def _path_to(self, wav):
        return os.path.join(self.path, "audio", "fold%s" % (wav[1],), wav[0])
    def _path_to_batch(self, idx):
        path_mfcc = "tmp/batch-size-%s-noise-%s-%s-pitch-%s-%s-bins-%s/%s-mfcc.npy" % (self.batch_size,
                                                                                       params.NOISE_LEVELS,
                                                                                       params.NOISE_PATTERNS,
                                                                                       params.PITCH_LOWER,
                                                                                       params.PITCH_UPPER,
                                                                                       params.MFCC_BINS,
                                                                                       idx)
        path_tags = "tmp/batch-size-%s-noise-%s-%s-pitch-%s-%s-bins-%s/%s-tags.npy" % (self.batch_size,
                                                                                       params.NOISE_LEVELS,
                                                                                       params.NOISE_PATTERNS,
                                                                                       params.PITCH_LOWER,
                                                                                       params.PITCH_UPPER,
                                                                                       params.MFCC_BINS,
                                                                                       idx)
        return path_mfcc, path_tags
    def __getitem__(self, idx):
        path_mfcc, path_tags = self._path_to_batch(idx)
        self._get_all()
        if self._cache[idx] is None:
            self._cache[idx] = numpy.load(path_mfcc, mmap_mode='r'), numpy.load(path_tags, mmap_mode='r')
        return self._cache[idx]
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(params.MFCC_BINS, params.MFCC_CNN_DEPTH,1), activation="relu"))
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(Conv2D(16, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    # model.add(Dense(10))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(optimizer='adadelta',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

def lstm_model():
    model = Sequential()
    model.add(LSTM(params.MFCC_CNN_DEPTH, input_shape=(params.MFCC_BINS, 10)))
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
    x_test, y_test = common.load_data(fold_selector=lambda r: r[common.FOLD] == "10", augment=False)
    x_test = numpy.stack(list(map(common.add_dim, x_test)), axis=0)
    y_test = numpy.stack(y_test)
    model.fit_generator(generator=train_gen,
              validation_data=(x_test, y_test),
              steps_per_epoch=len(train_gen),
              use_multiprocessing=False,
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
