#!/usr/bin/python3

data_imsize = (100, 100)
filtersize = (32, 32)

assert data_imsize[0] == data_imsize[1]
assert filtersize[0] == filtersize[1]
scale_factor = data_imsize[0] / filtersize[0]

filenames_mmapped = {name: '{}.mmapped'.format(name) for name in
                     ['xtrain', 'ytrain', 'xval', 'yval', 'xtest', 'ytest']}

default_figsize = (16, 8)
