#!/usr/bin/python3

data_imsize = (100, 100)
filtersize = (32, 32)

filenames_mmapped = {name: '{}.mmapped'.format(name) for name in
                     ['xtrain', 'ytrain', 'xval', 'yval', 'xtest', 'ytest']}

default_figsize = (16, 8)
