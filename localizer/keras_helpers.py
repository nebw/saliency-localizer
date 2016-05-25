#!/usr/bin/python3

import contextlib
import json
import os
from os.path import isfile

import numpy as np
import keras.backend as K
from keras.optimizers import SGD
from keras.objectives import mse
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import generic_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.layers.core import Flatten
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import theano


from localizer import visualization


def get_datagen(X):
    datagen = ImageDataGenerator(
        featurewise_center=False,
        featurewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False,
        vertical_flip=False)

    Xsample = X[np.random.choice(X.shape[0], 10000), :]
    datagen.fit(Xsample)

    return datagen

class HistoryCallback(Callback):
    def on_train_begin(self, logs={}):
        self.batch_hist = []
        self.epoch_hist = []

    def on_batch_end(self, batch, logs={}):
        self.batch_hist.append((logs.get('loss'), logs.get('acc')))

    def on_epoch_end(self, batch, logs={}):
        self.epoch_hist.append((logs.get('val_loss'), logs.get('val_acc')))

class AutomaticLearningRateScheduler(Callback):
    def __init__(self, optimizer, metric, min_improvment=0.001,
                 epoch_patience=3, factor=0.25, verbose=False):
        assert hasattr(optimizer, 'lr')
        self.optimizer = optimizer
        self.metric = metric
        self.current_best = np.infty
        self.current_best_epoch = 0
        self.min_improvment = min_improvment
        self.epoch_patience = epoch_patience
        self.epoch_log = []
        self.factor = factor
        self.verbose = verbose

    def on_train_begin(self, logs={}):
        self.current_best = np.infty
        self.current_best_epoch = 0

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_log = []

    def on_batch_end(self, batch, logs={}):
        self.epoch_log.append(logs[self.metric])

    def on_epoch_end(self, epoch, logs={}):
        mean_loss = np.array(self.epoch_log).mean()
        if mean_loss + self.min_improvment <= self.current_best:
            self.current_best = mean_loss
            self.current_best_epoch = epoch

        if self.verbose:
            print()
            print('{} epochs since last improvement'.format(epoch - self.current_best_epoch))

        if epoch - self.current_best_epoch > self.epoch_patience:
            lr = K.get_value(self.optimizer.lr)
            new_lr = lr*self.factor
            self.min_improvment *= self.factor
            K.set_value(self.optimizer.lr, new_lr)
            if not self.verbose:
                print()
            print("Reduce learning rate to: {:08f}".format(new_lr))
            self.current_best_epoch = epoch

from tempfile import mkdtemp
import os.path as path

def fit_model(model, datagen, X_train, y_train, X_test, y_test, weight_path, class_weight,
              nb_epoch=100, batchsize=4096, categorial=True):
    show_accuracy = y_train.shape[1] != 1
    metrics = ['accuracy'] if show_accuracy else []

    optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
    model.compile(optimizer, mse, metrics=metrics)
    checkpointer = ModelCheckpoint(filepath=weight_path, verbose=0, save_best_only=True)
    stopper = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    history = HistoryCallback()
    scheduler = AutomaticLearningRateScheduler(optimizer, 'loss', min_improvment=0.0001, verbose=True)

    callbacks = [checkpointer, stopper, history, scheduler]

    for callback in callbacks:
        callback._set_model(model)
        callback.on_train_begin()

    model.stop_training = False

    def unpack_result(res, show_accuracy, stage='train'):
        if show_accuracy:
            loss, acc = res
        else:
            loss = res
            acc = None

        progbarvals = [("{} loss".format(stage), loss)]
        if acc is not None:
            progbarvals.append(("{} acc".format(stage), acc))

        return loss, acc, progbarvals

    @contextlib.contextmanager
    def shuffled_mmap(data, fname, indices):
        try:
            filename = path.join(mkdtemp(), fname)
            fp = np.memmap(filename, dtype=data.dtype, mode='w+', shape=data.shape)
            np.copyto(fp, data[indices])
            yield fp
        finally:
            del(fp)
            os.remove(filename)

    for e in range(nb_epoch):
        print('Epoch', e)

        for callback in callbacks:
            callback.on_epoch_begin(e, None)

        indices_tr = np.random.permutation(X_train.shape[0])
        indices_te = np.random.permutation(X_test.shape[0])

        with shuffled_mmap(X_train, 'Xtr', indices_tr) as Xtr, \
             shuffled_mmap(y_train, 'ytr', indices_tr) as ytr, \
             shuffled_mmap(X_test, 'Xte', indices_te) as Xte, \
             shuffled_mmap(y_test, 'yte', indices_te) as yte:

            progbar = generic_utils.Progbar(Xtr.shape[0] + batchsize)
            for batches, (X_batch, Y_batch) in enumerate(
                    datagen.flow(Xtr, ytr, batch_size=batchsize, shuffle=False)):
                loss, acc, pbval = unpack_result(model.train_on_batch(X_batch, Y_batch,
                                                                    class_weight=class_weight),
                                                 show_accuracy)

                logs = {'loss': loss, 'acc': acc}
                for callback in callbacks:
                    callback.on_batch_end(None, logs)

                progbar.add(batchsize, values=pbval)

                if batches >= Xtr.shape[0] / batchsize:
                    break

            num_test  = 0
            mean_loss = 0.
            mean_acc  = 0.
            progbar = generic_utils.Progbar(X_test.shape[0] + batchsize)
            for batches, (X_batch, Y_batch) in enumerate(
                    datagen.flow(Xte, yte, batch_size=batchsize, shuffle=False)):
                loss, acc, pbval = unpack_result(model.test_on_batch(X_batch, Y_batch),
                                                 show_accuracy, stage='test')

                num_test += 1
                mean_loss += loss
                if show_accuracy:
                    mean_acc += acc

                progbar.add(batchsize, values=pbval)

                if batches >= Xte.shape[0] / batchsize:
                    break

            mean_loss /= num_test
            if show_accuracy:
                mean_acc /= num_test

            logs = {'val_loss': mean_loss}
            if acc is not None:
                logs['val_acc'] = mean_acc
            for callback in callbacks:
                callback.on_epoch_end(e, logs)

            if model.stop_training:
                break

            print()

    model.load_weights(weight_path)

    return history

def predict_model(model, X, datagen):
    y = np.zeros((X.shape[0], 2))
    cnt = 0
    batchsize = 256
    progbar = generic_utils.Progbar(X.shape[0] + batchsize)
    for X_batch, _ in datagen.flow(X, y, batch_size=batchsize):
        y_batch = model.predict(X_batch, batch_size=batchsize, verbose=0)
        # conversion from singular output to categorial
        if y_batch.shape[1] == 1:
            y[cnt:min(X.shape[0], cnt+batchsize), 0] = 1. - y_batch[:, 0]
            y[cnt:min(X.shape[0], cnt+batchsize), 1] = y_batch[:, 0]
        else:
            y[cnt:min(X.shape[0], cnt+batchsize), :] = y_batch
        cnt += batchsize
        progbar.add(batchsize, values=[])

        if cnt >= y.shape[0]:
            break
    return y

def restore_or_fit_model(model, datagen, Xtrain, ytrain, Xtest, ytest, weight_path, class_weight,
                         nb_epoch=20, batchsize=4096):
    if isfile(weight_path):
        model.load_weights(weight_path)
    else:
        return fit_model(model, datagen, Xtrain, ytrain, Xtest, ytest, weight_path, class_weight,
                         nb_epoch, batchsize)

def f_score(precision, recall, b = 1):
    b2 = np.power(b, 2)
    return (1 + b2) * (precision * recall) / ((b2 * precision) + recall)

def evaluate_model(ytest, ytest_output, visualize=False, ax=None):
    assert(ytest.ndim <= 2)
    assert(ytest_output.ndim <= 2)
    if ytest.ndim == 2 and ytest.shape[1] == 2:
        ytest = ytest[:, 1]
    if ytest_output.ndim == 2 and ytest_output.shape[1] == 2:
        ytest_output = ytest_output[:, 1]
    fpr, tpr, _ = roc_curve(ytest, ytest_output)
    roc_auc    = auc(fpr, tpr)

    precision, recall, thresholds = precision_recall_curve(ytest, ytest_output)
    average_precision = average_precision_score(ytest, ytest_output)

    assert(len(thresholds) >= 2)

    if visualize:
        visualization.plot_roc(fpr, tpr, roc_auc, ax)
        visualization.plot_recall_precision(recall, precision, average_precision, ax)

    return precision, recall, average_precision, thresholds, fpr, tpr, roc_auc

def store_evaluation_results(path, precision, recall, average_precision, thresholds, fpr, tpr, roc_auc):
    data = {'precision': precision, 'recall': recall, 'average_precision': average_precision,
            'thresholds': thresholds, 'fpr': fpr, 'tpr': tpr, 'roc_auc': roc_auc}
    with open('evaluation_results.json', 'w') as outfile:
        json.dumps(data, outfile, sort_keys=True)

def load_evaluation_results(path):
    if not isfile(path):
        raise ValueError('Invalid path')

    with open('evaluation_results.json', 'r') as infile:
        return json.loads(infile)

def select_threshold(evaluation_results, min_value, optimize='recall'):
    return select_threshold(evaluation_results['precision'], evaluation_results['recall'],
                            evaluation_results['thresholds'], min_value, optimize)

def select_threshold(precision, recall, thresholds, min_value, optimize='recall'):
    assert(optimize in ['precision', 'recall'])
    if optimize == 'precision':
        selected_index = np.nonzero(precision >= min_value)[0][0]
    elif optimize == 'recall':
        selected_index = np.nonzero(recall >= min_value)[0][-1]

    if selected_index >= len(thresholds):
        print('Value too high!')
        selected_index = len(thresholds) - 2

    if (selected_index == len(thresholds) - 1):
        selected_index = len(thresholds) - 2

    selected_precision = precision[selected_index]
    selected_recall    = recall[selected_index]
    selected_threshold = thresholds[selected_index]

    print(('Recall', selected_recall))
    print(('Precision', selected_precision))
    print(('Threshold', selected_threshold))

    print(('F_2', f_score(selected_precision, selected_recall, b=2)))
    print(('F_0.5', f_score(selected_precision, selected_recall, b=0.5)))

    return selected_threshold

def copy_network_weights(train, deploy):
    for tr_layer, dp_layer in zip(train.layers, deploy.layers):
        if type(tr_layer) == Flatten:
            continue
        for tp, dp in zip(tr_layer.trainable_weights,
                          dp_layer.trainable_weights):
            dp.set_value(tp.get_value())

def get_convolution_function(train_model, convolution_model):
    copy_network_weights(train_model, convolution_model)

    mode = theano.compile.get_default_mode()
    fun = theano.function([convolution_model.inputs[0], K.learning_phase()],
                           convolution_model.outputs,
                           mode=mode)

    # 0 => learning_phase = test phase
    return lambda x : fun(x, 0)

def filter_by_threshold(X, Xs, y, threshold, network, datagen, prop_below=1.):
    y_out = predict_model(network, Xs, datagen)
    above_indices = np.nonzero(y_out[:, 1] > threshold)[0]
    below_indices = np.random.choice((np.nonzero(y_out[:, 1] <= threshold))[0],
                                     prop_below * above_indices.shape[0],
                                     replace=False)
    filter_indices = np.concatenate((above_indices, below_indices))
    Xf = np.copy(X[filter_indices, :])
    yf = np.copy(y[filter_indices, :])
    return Xf, yf
