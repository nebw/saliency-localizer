#!/usr/bin/python3

data_imsize = (64, 64)
filtersize = (16, 16)

filenames_mmapped = {name: '{}.mmapped'.format(name) for name in
                     ['xtrain', 'ytrain', 'xval', 'yval', 'xtest', 'ytest']}

default_figsize = (16, 8)
