#!/usr/bin/python3

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import theano.tensor as T
from keras.objectives import epsilon, mse


def get_filter_network():
    model = Sequential()

    model.add(Convolution2D(16, 1, 2, 2, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 16, 2, 2, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 16, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(32, 16, 2, 2, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 32, 2, 2, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 32, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Convolution2D(64, 32, 2, 2, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 64, 2, 2, border_mode='full'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 64, 2, 2))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(poolsize=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense((8 * 8) * 64, 1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, 1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, 1))
    model.add(Activation('sigmoid'))

    model.compile(loss='mse', optimizer="adam")

    return model

def get_saliency_network(train=True):
    model = Sequential()

    model.add(Convolution2D(32, 1, 3, 3, activation='relu', border_mode='same'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(128, 32, 16, 16, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(1, 128, 1, 1, activation='sigmoid'))

    if train:
        model.add(Flatten())

    model.compile('adam', mse)

    return model