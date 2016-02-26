#!/usr/bin/python3

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.objectives import mse
from localizer.config import data_imsize, filtersize
from keras.optimizers import SGD


def get_filter_network():
    model = Sequential()

    model.add(Convolution2D(8, 2, 2,
                            input_shape=(1, data_imsize[0], data_imsize[1]),
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(8, 2, 2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Convolution2D(16, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(16, 2, 2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Convolution2D(32, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, 2, 2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('softmax'))
    model.add(BatchNormalization())

    model.compile(loss='categorial_crossentropy', optimizer="adam")

    return model


def get_saliency_network(train=True, learning_rate=0.01, shape=None):
    if train:
        input_shape = (1, filtersize[0], filtersize[1])
    else:
        input_shape = (1, shape[0], shape[1])

    model = Sequential()
    model.add(Convolution2D(32, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2), input_shape=input_shape))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(128, 5, 5, activation='relu', border_mode='valid'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(1, 1, 1, activation='sigmoid'))

    if train:
        model.add(Flatten())
    else:
        model.add(UpSampling2D())
        model.add(UpSampling2D())

    model.compile(SGD(lr=learning_rate, momentum=0.9, nesterov=True), mse)

    return model
