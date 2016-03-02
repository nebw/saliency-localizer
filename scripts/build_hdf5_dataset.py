#! /usr/bin/env python

import localizer.config
from localizer.util import extract_rois
from scipy.misc import imread
import h5py
import keras
import numpy as np
import json
import argparse
import os


def write_cache_to_hdf5(cache, h5, pos):
    nb_samples = sum([len(b) for b in cache['rois']])
    h5_end = pos + nb_samples
    permutation = np.random.permutation(nb_samples)

    def write_dataset(dataset, data):
        data = np.concatenate(data)
        assert len(data) == nb_samples
        dataset.resize(h5_end, axis=0)
        dataset[pos:h5_end] = data[permutation]

    write_dataset(h5['tags'], cache['rois'])
    write_dataset(h5['saliencies'], cache['saliencies'])
    h5.flush()
    return h5_end


def clear_cache():
    return {'rois': [], 'saliencies': []}


def main(json_file, hdf5_fname, roi_size, image_dir, threshold, offset):
    tag_positions = json.load(json_file)
    assert not os.path.exists(hdf5_fname), \
        "hdf5 file already exists: {}".format(hdf5_fname)
    h5 = h5py.File(hdf5_fname)
    nb_chunks = 1024
    roi_shape = (roi_size, roi_size)
    h5.create_dataset("tags",
                      shape=(1, 1, roi_shape[0], roi_shape[1]),
                      maxshape=(None, 1, roi_shape[0], roi_shape[1]),
                      dtype='uint8', compression='gzip',
                      chunks=(nb_chunks, 1, roi_shape[0], roi_shape[1]))
    h5.create_dataset("saliencies",
                      shape=(1,),
                      maxshape=(None,),
                      dtype='float32', compression='gzip',
                      chunks=(nb_chunks, ))
    h5_pos = 0
    cache = clear_cache()
    progbar = keras.utils.generic_utils.Progbar(len(tag_positions["images"]))
    nb_batches = 96
    for i, im_json in enumerate(tag_positions['images']):
        try:
            image = imread(im_json['filename'])
        except OSError as e:
            print(e)
            continue
        candidates = np.array(im_json['candidates'])
        saliency = np.array(im_json['saliencies']).reshape((-1))
        selection = saliency >= threshold
        big_enough_candidates = candidates[selection] + offset
        rois = extract_rois(big_enough_candidates, image, roi_shape)
        cache['rois'].append(rois)
        cache['saliencies'].append(saliency[selection])
        if len(cache['rois']) > nb_batches:
            h5_pos = write_cache_to_hdf5(cache, h5, h5_pos)
            cache = clear_cache()
        progbar.update(i+1)

    write_cache_to_hdf5(cache, h5, h5_pos)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Find the tags in many images and saves the position in a'
        ' json file.')
    parser.add_argument('-o', '--out', type=str,
                        default='tags.hdf5',
                        help='hdf5 file where the tags will be saved')
    parser.add_argument('-r', '--roi-size', type=int,
                        default=localizer.config.data_imsize[0],
                        help='directory with the saliency network weights')
    parser.add_argument('--offset', type=int, default=0,
                        help="offset that is added to the tag's coordinates")
    parser.add_argument('-t', '--threshold', type=float, default=0,
                        help='threshold only tags above will be selected.')
    parser.add_argument('-i', '--image-dir', type=str,
                        help='images will be used from this directory.')
    parser.add_argument('input', type=argparse.FileType('r'),
                        help='json file from `find_tags.py`')
    args = parser.parse_args()
    main(args.input, args.out, args.roi_size, args.image_dir,
         args.threshold, args.offset)
