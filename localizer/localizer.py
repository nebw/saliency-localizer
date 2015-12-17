#!/usr/bin/python3

from os.path import isfile, join, splitext, exists, isdir

from localizer import models, util, keras_helpers, config


class Localizer:
    def __init__(self, data_dir, logger='default'):
        if logger == 'default':
            self.logger = util.get_default_logger()
        else:
            self.logger = logger

        if not isdir(data_dir):
            raise ValueError('data_dir ist not a valid directory')

        subdirs = util.get_subdirectories(data_dir)
        if 'weights' in subdirs:
            saliency_weight_file = join(data_dir, 'weights', 'saliency-weights')
            self.logger.info('Loading saliency network model...')
            saliency_network = models.get_saliency_network(train=True)
            if isfile(saliency_weight_file):
                self.logger.info('Restoring saliency network weights...')
                saliency_network.load_weights(saliency_weight_file)
            else:
                self.logger.info('Training saliency network...')
                pass
                # TODO: train saliency model

            self.logger.info('Compiling saliency network convolution function...')
            saliency_conv_model = models.get_saliency_network(train=False)
            self.convolution_function = keras_helpers.get_convolution_function(saliency_network, saliency_conv_model)

            filter_weight_file = join(data_dir, 'weights', 'filter_weights')
            self.logger.info('Loading filter network model...')
            self.filter_network = models.get_filter_network()
            if isfile(filter_weight_file):
                self.logger.info('Restoring filter network weights...')
                self.filter_network.load_weights(filter_weight_file)
            else:
                self.logger.info('Training filter network...')
                pass
                # TODO: train filter model
        # TODO: load or reevaluate thresholds

    def detect_tags(self, image_path, saliency_threshold):
        # TODO: default value for saliency threshold based on evaluation
        image, image_filtersize, targetsize = util.preprocess_image(image_path, config.filtersize)

        saliency = self.convolution_function(
            image_filtersize.reshape((1, 1, image_filtersize.shape[0], image_filtersize.shape[1])))

        candidates = util.get_candidates(saliency, saliency_threshold)
        rois, saliencies = util.extract_rois(candidates, saliency, image)

        # TODO: filter candidates using filter network

        return saliencies, candidates, rois
