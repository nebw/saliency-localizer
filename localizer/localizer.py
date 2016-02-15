#!/usr/bin/python3

from os.path import isfile, join, isdir
from localizer import models, util, keras_helpers, config


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
        self.filter_network = models.get_filter_network(compile=False)

    def load_weights(self, data_dir):
        if not isdir(data_dir):
            raise ValueError('data_dir ist not a valid directory')
        saliency_weight_file = join(data_dir, 'saliency-weights')
        if isfile(saliency_weight_file):
            self.logger.info('Restoring saliency network weights...')
            self.saliency_network.load_weights(saliency_weight_file)

        filter_weight_file = join(data_dir, 'filter_weights')
        if isfile(filter_weight_file):
            self.logger.info('Restoring filter network weights...')
            self.filter_network.load_weights(filter_weight_file)

    def train_saliency(self):
        self.logger.info('Training saliency network...')
        raise NotImplemented()

    def train_filter(self):
        self.logger.info('Training filter network...')
        # TODO: load or reevaluate thresholds
        raise NotImplemented()

    def compile(self):
        self.logger.info('Compiling saliency network convolution function...')
        saliency_conv_model = models.get_saliency_network(train=False)
        self.convolution_function = keras_helpers.get_convolution_function(
            self.saliency_network, saliency_conv_model)

    def detect_tags(self, image_path, saliency_threshold):
        # TODO: default value for saliency threshold based on evaluation
        image, image_filtersize, targetsize = util.preprocess_image(
            image_path, config.filtersize)

        saliency = self.convolution_function(
            image_filtersize.reshape((1, 1, image_filtersize.shape[0],
                                      image_filtersize.shape[1])))

        candidates = util.get_candidates(saliency, saliency_threshold)
        rois, saliencies = util.extract_rois(candidates, saliency, image)

        # TODO: filter candidates using filter network

        return saliencies, candidates, rois
