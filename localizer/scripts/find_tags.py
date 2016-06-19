#! /usr/bin/env python

from localizer.localizer import Localizer
from localizer.util import get_default_logger
from scipy.misc import imread
import os
import keras
from random import shuffle
import json
import time
import argparse


def run(image_path_file, network_weights, json_file, threshold):
    log = get_default_logger()
    beesbook_images = [l.rstrip('\n') for l in image_path_file.readlines()]
    for imfname in beesbook_images:
        assert os.path.isabs(imfname), \
            "image `{}` is not absolute or does not exists".format(imfname)
    shuffle(beesbook_images)

    loc = Localizer()
    loc.load_weights(network_weights)
    image_shape = imread(beesbook_images[0]).shape
    log.info("Image shape is {}".format(image_shape))
    loc.compile(image_shape=image_shape)

    progbar = keras.utils.generic_utils.Progbar(len(beesbook_images))
    images = []
    try:
        for i, imfname in enumerate(beesbook_images):
            image = {
                "filename": imfname
            }

            saliencies, candidates, _, _ = loc.detect_tags(imfname, threshold)
            image['candidates'] = candidates.tolist()
            image['saliencies'] = saliencies.tolist()
            images.append(image)
            progbar.add(1)
    finally:
        json_obj = {
            "time": time.time(),
            "threshold": threshold,
            "images": images,
            "image_shape": image_shape
        }
        json.dump(json_obj, json_file)


def main():
    parser = argparse.ArgumentParser(
        description='Find the tags in many images and saves the position in a'
        ' json file.')
    parser.add_argument('-o', '--out', type=argparse.FileType('w+'),
                        default='tag_positions.json',
                        help='json file with the tag positions')
    parser.add_argument('-w', '--weights', type=str,
                        required=True,
                        help='weights of the saliency network')
    parser.add_argument('-t', '--threshold', type=float,
                        required=True,
                        help='threshold for the saliency network. '
                        'Should be somewhere between 0.4 and 0.6')
    parser.add_argument('images', type=argparse.FileType('r'),
                        help='file with one image filename per line.')
    arg = parser.parse_args()

    run(arg.images, arg.weights, arg.out, arg.threshold)


if __name__ == "__main__":
    main()
