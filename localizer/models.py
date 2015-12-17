#!/usr/bin/python3

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.objectives import mse
from localizer.config import data_imsize


def get_filter_network():
    model = Sequential()

    model.add(Convolution2D(16, 2, 2,
                            input_shape=(1, data_imsize[0], data_imsize[1]),
                            border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Convolution2D(32, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 2, 2, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer="adam")

    return model


def get_saliency_network(train=True):
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, input_shape=(1, None, None),
                            activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 16, 16, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(1, 1, 1, activation='sigmoid'))

    if train:
        model.add(Flatten())

    model.compile('adam', mse)

    return model