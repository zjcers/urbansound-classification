import itertools
import random
import os
import soundfile
import csv
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
# from hyperas.distributions import choice, uniform
# from hyperopt import STATUS_OK
import params
# from hyperopt import Trials, tpe
# import hyperas.optim


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

class DataGenerator:
    """
    """
    class Rows:
        SLICE_FILE_NAME = 0
        FOLD = 5
        CLASS_ID = 6
    def __init__(self, batch_size, test_fold, validation=False, path="UrbanSound8K"):
        self.batch_size = batch_size
        self.test_fold = test_fold
        self.validation = validation
        self.path = path
    @staticmethod
    def _multiplication_factor():
        return (params.NOISE_LEVELS + 1) \
             * (params.PITCH_UPPER - params.PITCH_LOWER + 1)
    @staticmethod
    def _slices_per_wav(filename):
        with soundfile.SoundFile(filename) as wav:
            mfcc_length = len(wav) // common.ms_to_samples(wav.samplerate, params.MFCC_WINDOW_LENGTH)
            slices = len(range(0, mfcc_length, params.MFCC_CNN_DEPTH // params.MFCC_OVERLAP_FACTOR))
            slices *= wav.channels
            slices *= DataGenerator._multiplication_factor()
            return slices
    def _list_wavs(self, for_train = True):
        with open(os.path.join(self.path, "metadata", "UrbanSound8K.csv"), 'r') as metadata:
            reader = csv.reader(metadata)
            next(reader) # skip header
            for row in reader:
                folds_eq = row[self.Rows.FOLD] == self.test_fold
                if (for_train and not folds_eq) or (not for_train and folds_eq):
                    yield (row[self.Rows.SLICE_FILE_NAME], row[self.Rows.FOLD], int(row[self.Rows.CLASS_ID]))
    def __len__(self):
        total_slices = 0
        for wav in self._list_wavs():
            path = self._path_to(wav)
            total_slices += DataGenerator._slices_per_wav(path)
        return total_slices // self.batch_size
    def _per_epoch(self):
            for wav in self._list_wavs(not self.validation):
                if self.validation:
                    y, sr = common.load_wav_mono(self._path_to(wav))
                    mfcc = common.process_mfcc(y, sr)
                    for slice in common.mfcc_reshape(mfcc):
                        yield slice, wav[2]
                else:
                    for sr, augment_data in common.augment(self._path_to(wav)):
                        mfcc = common.process_mfcc(augment_data, sr)
                        for slice in common.mfcc_reshape(mfcc):
                            yield slice, wav[2]
    def _path_to(self, wav):
        return os.path.join(self.path, "audio", "fold%s" % (wav[1],), wav[0])

    def __getitem__(self, idx):
        path_mfcc = "tmp/batch/%s-mfcc.npy" % (idx,)
        path_tags = "tmp/batch/%s-tags.npy" % (idx,)
        if os.path.exists(path_mfcc):
            return numpy.load(path_mfcc), numpy.load(path_tags)
        batch_inputs = []
        batch_targets = []
        while True:
            for slice, target in self._per_epoch():
                batch_inputs.append(slice)
                batch_targets.append(target)
                if len(batch_inputs) == self.batch_size:
                    targets = keras.utils.to_categorical(batch_targets, 10)
                    yield numpy.stack(batch_inputs, axis=0), targets
                    batch_inputs = []
                    batch_targets = []

def cnn_model(x_train, y_train, x_test, y_test):
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
    model.add(Dense(40))
    model.add(Dense(10))
    model.add(Activation("softmax"))
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    checkpoints = keras.callbacks.ModelCheckpoint("model-cnn-{epoch:02d}.hdf5", save_best_only=True, period=2)
    earlystop = keras.callbacks.EarlyStopping(monitor="val_acc", min_delta=0.01, patience=3, restore_best_weights=True)
    tensorboard = keras.callbacks.TensorBoard(log_dir="./logs-cnn")
    BATCH_SIZE = 128
    train_gen = DataGenerator(BATCH_SIZE, "10")
    valid_gen = DataGenerator(BATCH_SIZE, "10", True)
    # model.fit_generator(generator=train_gen,
    #           validation_data=common.load_data(lambda r: r[common.FOLD] == "10"),
    #           steps_per_epoch=len(train_gen),
    #           epochs=200)
    model.fit(x=x_train,
              y=y_train,
              batch_size=128,
              callbacks=[checkpoints, earlystop, tensorboard],
              epochs=200,
              validation_data=(x_test, y_test))
    return model

# def data_wrapper():
#     return common.get_data()

# best_run, best_model = hyperas.optim.minimize(model=cnn_model,
#                                               data=data_wrapper,
#                                               algo=tpe.suggest,
#                                               max_evals=5,
#                                               trials=Trials())
# print("Evaluation of the best model:")
# x_test, y_test = common.load_data(fold="1")
x_train, y_train, x_test, y_test = common.get_data()
model = cnn_model(x_train, y_train, x_test, y_test)
model.save("urbansound.hdf5")
