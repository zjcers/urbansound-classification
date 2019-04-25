import itertools
import random
from keras.utils import to_categorical
import keras
import numpy
import common
import noise
import params
from model import model


common.print_header("Loading Data")
labels, data = common.load_data()

print("Loaded", len(data),"items")

common.print_header("Calculating category vectors")
category_vecs = to_categorical(labels)

# Squish all of the MFCCs together into one numpy array
def add_dim(arr: numpy.ndarray):
    old_shape = arr.shape
    return arr.reshape(old_shape[0], old_shape[1], 1)
xs = numpy.stack(list(map(add_dim, data)), axis=0)
#xs = numpy.stack(data, axis=0)
print("Input data shape:", xs.shape)

model.save("urbansound.hdf5")
