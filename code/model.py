from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import MaxPooling2D
import params

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(params.MFCC_BINS, params.MFCC_CNN_DEPTH,1), activation="relu"))
model.add(Conv2D(32, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(Conv2D(16, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

