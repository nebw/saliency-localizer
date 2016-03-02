#! /usr/bin/env python

from localizer.localizer import Localizer
from localizer.util import get_default_logger, to_image_coordinates
from os.path import join
from scipy.misc import imread
import os
import keras
from random import shuffle
import json
import time
import hashlib
import argparse


def sha1_of_file(filepath):
    with open(filepath, 'rb') as f:
        return hashlib.sha1(f.read()).hexdigest()


def main(image_path_file, network_weight_dir, json_file, threshold):
    log = get_default_logger()
    beesbook_images = [l.rstrip('\n') for l in image_path_file.readlines()]
    for imfname in beesbook_images:
        assert os.path.isabs(imfname), \
            "image `{}` is not absolute or does not exists".format(imfname)
    shuffle(beesbook_images)

    loc = Localizer()
    loc.load_weights(network_weight_dir)
    image_shape = imread(beesbook_images[0]).shape
    log.info("Image shape is %", )
    loc.compile(image_shape=image_shape)

    progbar = keras.utils.generic_utils.Progbar(len(beesbook_images))
    images = []
    try:
        for i, imfname in enumerate(beesbook_images):
            image = {
                "sha1": sha1_of_file(imfname),
                "filename": imfname
            }
            saliencies, candidates, _ = loc.detect_tags(imfname, threshold)
            image['candidates'] = candidates.tolist()
            image['saliencies'] = saliencies.tolist()
            images.append(image)
            progbar.add(1)
    finally:
        json_obj = {
            "time": time.time(),
            "threshold": threshold,
            "images": images,
        }
        json.dump(json_obj, json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Find the tags in many images and saves the position in a'
        ' json file.')
    parser.add_argument('-o', '--out', type=argparse.FileType('w+'),
                        default='tag_positions.json',
                        help='json file with the tag positions')
    parser.add_argument('-w', '--weight-dir', type=str,
                        required=True,
                        help='directory with the saliency network weights')
    parser.add_argument('-t', '--threshold', type=float,
                        required=True,
                        help='threshold for the saliency network. '
                        'Should be somewhere between 0.4 and 0.6')
    parser.add_argument('images', type=argparse.FileType('r'),
                        help='file with one image filename per line.')
    arg = parser.parse_args()
    print(arg)
    main(arg.images, arg.weight_dir, arg.out, arg.threshold)
