#!/usr/bin/python3

from os.path import isfile, join, isdir
from localizer import models, util, keras_helpers, config
from skimage.filters import gaussian_filter

import numpy as np


class Localizer:
    """
    The Localizer exists of two networks, the saliency network and the filter
    network. The saliency networks is fast and runs over the whole
    image as convolution.  The filter network is relative slow and
    only filters the results of the saliency network.
    """
    def __init__(self, logger='default'):
        if logger == 'default':
            self.logger = util.get_default_logger()
        else:
            self.logger = logger

        self.saliency_network = models.get_saliency_network(
            train=True, compile=False)

    def load_weights(self, data_dir):
        if not isdir(data_dir):
            raise ValueError('data_dir ist not a valid directory')
        saliency_weight_file = join(data_dir, 'saliency-weights')
        if isfile(saliency_weight_file):
            self.logger.info('Restoring saliency network weights...')
            self.saliency_network.load_weights(saliency_weight_file)
        else:
            raise ValueError('invalid weight file')

    def train_saliency(self):
        self.logger.info('Training saliency network...')
        raise NotImplemented()

    def compile(self, image_shape=(3060, 4060)):
        self.logger.info('Compiling saliency network convolution function...')
        ratio = config.filtersize[0] / config.data_imsize[0]
        targetsize = np.round(np.array(image_shape) * ratio).astype(int)
        saliency_conv_model = models.get_saliency_network(train=False, shape=targetsize)
        self.convolution_function = keras_helpers.get_convolution_function(
            self.saliency_network, saliency_conv_model)

    def detect_tags(self, image_path, saliency_threshold=0.5):
        image, image_filtersize, targetsize = util.preprocess_image(
            image_path, config.filtersize)

        saliency = self.convolution_function(
            image_filtersize.reshape((1, 1, image_filtersize.shape[0],
                                      image_filtersize.shape[1])))

        saliency[0][0, 0] = gaussian_filter(saliency[0][0, 0], sigma=2.)

        candidates = util.get_candidates(saliency, saliency_threshold)
        rois = util.extract_rois(candidates, image)
        saliencies = util.extract_saliencies(candidates, saliency)

        return saliencies, candidates, rois
