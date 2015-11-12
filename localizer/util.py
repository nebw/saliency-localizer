#!/usr/bin/python3

from os import listdir, makedirs, removedirs
from os.path import isfile, join, splitext, exists
import itertools
import shutil
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
from scipy.misc import imresize, imread
from sklearn.cross_validation import StratifiedShuffleSplit
from keras.utils import generic_utils

from localizer.config import data_imsize, filenames_mmapped


def get_hdf5_files(dname):
    return [join(dname, f) for f in listdir(dname)
            if isfile(join(dname, f))
            and splitext(f)[1] == '.hdf5']

def get_num_samples(files):
    nsamples = 0
    for f in files:
        with h5py.File(f, 'r') as dset:
            nsamples += len(dset['data'])
    return nsamples

def get_shapes(nsamples):
    X_shape = (nsamples, 1, data_imsize[0], data_imsize[1])
    y_shape = (nsamples, 2)

    return X_shape, y_shape

def load_data(dname, fname_X, fname_y):
    files = get_hdf5_files(dname)

    nsamples = get_num_samples(files)
    X_shape, y_shape = get_shapes(nsamples)

    X = np.memmap(fname_X, dtype='float32', mode='w+', shape=X_shape)
    y = np.memmap(fname_y, dtype='float32', mode='w+', shape=y_shape)

    idx = 0
    progbar = generic_utils.Progbar(nsamples)
    for f in files:
        with h5py.File(f, 'r') as dset:
            data = dset['data']
            labels = dset['labels']

            idx_start = idx

            Xbatch = np.zeros((len(data), 1, data_imsize[0], data_imsize[1]))
            ybatch = np.zeros((len(data), 2))
            for hdf5_idx in range(len(data)):
                im = data[hdf5_idx][0]

                Xbatch[hdf5_idx, 0, :, :] = im.astype(np.float) / 255.
                ybatch[hdf5_idx, labels[hdf5_idx]] = 1

                idx +=1

                if not(idx % np.power(2, 10)):
                    progbar.add(np.power(2, 10), values=[])

            random_order = np.random.permutation(len(data))

            X[idx_start:idx, 0, :, :] = Xbatch[random_order, 0, :, :]
            y[idx_start:idx, :] = ybatch[random_order, :]

    return X, y

def restore_data(dir, X_file, y_file, testval = False):
    files = get_hdf5_files(dir)
    nsamples = get_num_samples(files)
    if testval:
        nsamples /= 2
    X_shape, y_shape = get_shapes(nsamples)

    X = np.memmap(X_file, dtype='float32', shape=X_shape, mode='r+')
    y = np.memmap(y_file, dtype='float32', shape=y_shape, mode='r+')

    return X, y

def split_validation(data_dir, X_test, y_test):
    temp_dir = join(data_dir, 'temp')
    existing_tempdir = exists(temp_dir)
    if not existing_tempdir:
        makedirs(temp_dir)

    filenames_test = [join(data_dir, f) for f in (filenames_mmapped['xtest'], filenames_mmapped['ytest'])]
    filenames_val = [join(data_dir, f) for f in (filenames_mmapped['xval'], filenames_mmapped['yval'])]
    filenames_temp = [join(temp_dir, f) for f in (filenames_mmapped['xtest'], filenames_mmapped['ytest'])]

    validation_indices, test_indices = next(StratifiedShuffleSplit(y_test[:, 0], 1, test_size=0.5)._iter_indices())

    X_shape_validation, y_shape_validation = get_shapes(validation_indices.shape[0])

    X_validation = np.memmap(filenames_val[0], dtype='float32', mode='w+', shape=X_shape_validation)
    y_validation = np.memmap(filenames_val[1], dtype='float32', mode='w+', shape=y_shape_validation)

    X_validation[:] = X_test[validation_indices]
    y_validation[:] = y_test[validation_indices]

    X_shape_test, y_shape_test = get_shapes(test_indices.shape[0])

    X_test_tmp = np.memmap(filenames_temp[0], dtype='float32', mode='w+', shape=X_shape_test)
    y_test_tmp = np.memmap(filenames_temp[1], dtype='float32', mode='w+', shape=y_shape_test)

    X_test_tmp[:] = X_test[test_indices]
    y_test_tmp[:] = y_test[test_indices]

    del(X_test)
    del(y_test)
    del(X_test_tmp)
    del(y_test_tmp)

    shutil.move(filenames_temp[0], filenames_test[0])
    shutil.move(filenames_temp[1], filenames_test[1])

    X_test = np.memmap(filenames_test[0], dtype='float32', mode='r+', shape=X_shape_test)
    y_test = np.memmap(filenames_test[1], dtype='float32', mode='r+', shape=y_shape_test)

    if not existing_tempdir:
        removedirs(temp_dir)

    return X_test, y_test, X_validation, y_validation

def iterative_shuffle(X, y, batchsize=np.power(2, 17)):
    idx = 0
    progbar = generic_utils.Progbar(X.shape[0])
    while idx < X.shape[0]:
        to_idx = min(X.shape[0], idx+batchsize)

        random_order = np.random.permutation(to_idx - idx)

        Xbatch = X[idx:to_idx]
        ybatch = y[idx:to_idx]

        X[idx:to_idx] = Xbatch[random_order, :, :, :]
        y[idx:to_idx] = ybatch[random_order, :]

        progbar.add(to_idx - idx, values=[])
        idx += batchsize

def load_or_restore_data(data_dir):
    train_dir = join(data_dir, 'train')
    test_dir = join(data_dir, 'test')

    filenames_train = [join(data_dir, f) for f in (filenames_mmapped['xtrain'], filenames_mmapped['ytrain'])]
    filenames_test = [join(data_dir, f) for f in (filenames_mmapped['xtest'], filenames_mmapped['ytest'])]
    filenames_val = [join(data_dir, f) for f in (filenames_mmapped['xval'], filenames_mmapped['yval'])]

    if all([isfile(f) for f in itertools.chain(filenames_train, filenames_test, filenames_val)]):
        print('Restoring mmapped data')
        X_train, y_train = restore_data(train_dir, *filenames_train)
        X_test, y_test = restore_data(test_dir, *filenames_test, testval=True)
        X_val, y_val = restore_data(test_dir, *filenames_val, testval=True)
    else:
        print('Loading data')
        X_train, y_train = load_data(train_dir, *filenames_train)
        X_test, y_test = load_data(train_dir, *filenames_test)
        print('')

        print('Shuffling data')
        iterative_shuffle(X_train, y_train)
        iterative_shuffle(X_test, y_test)
        print('')

        print('Splitting validation')
        X_test, y_test, X_val, y_val = split_validation(data_dir, X_test, y_test)
        print('')

    return (X_train, y_train, X_test, y_test, X_val, y_val)

def resize_data(X, targetsize, interp='bicubic'):
    f = NamedTemporaryFile(delete=False)
    f.close()

    Xout = np.memmap(f.name, dtype='float32', mode='w+', shape=(X.shape[0], 1, targetsize[0], targetsize[1]))

    progbar = generic_utils.Progbar(X.shape[0])
    for idx in range(X.shape[0]):
        Xout[idx, 0, :, :] = imresize(X[idx, 0, :, :], targetsize, interp=interp) / 255.

        if not(idx % np.power(2, 10)):
            progbar.add(np.power(2, 10), values=[])

    return Xout

from scipy.misc import imread, imresize

def preprocess_image(image_path, filter_imsize):
    assert(filter_imsize[0] == filter_imsize[1])
    ratio = filter_imsize[0] / data_imsize[0]

    image = imread(image_path)

    targetsize = np.round(np.array(image.shape) * ratio).astype(int)
    image_filtersize = imresize(image, targetsize, interp='bicubic')

    image_filtersize = np.lib.pad(image_filtersize, (int(filter_imsize[0] / 4), int(filter_imsize[1] / 4)), 'edge')
    targetsize = image_filtersize.shape

    return image, image_filtersize, targetsize