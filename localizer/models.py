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

    model.add(Dense((8 * 8) * 64, 1024, init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, 1024, init='glorot_uniform'))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1024, 2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer="adam")

    return model

def saliency_error(y_true, y_pred):
    y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
    base_loss = T.nnet.binary_crossentropy(y_pred, y_true)
    loss = T.set_subtensor(base_loss[T.eq(y_true, 1)],
                           base_loss[T.eq(y_true, 1)], inplace=False)
    return loss.mean(axis=-1)


def get_saliency_network(train=True):
    model = Sequential()

    model.add(Convolution2D(4, 1, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(8, 4, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(16, 8, 3, 3, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Convolution2D(32, 16, 10, 10, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(1, 32, 1, 1, activation='sigmoid'))

    if train:
        model.add(Flatten())

    model.compile('adam', mse)

    return model