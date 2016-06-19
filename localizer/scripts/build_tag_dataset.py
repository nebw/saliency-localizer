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
import queue
import threading


def write_cache_to_hdf5(cache, h5, pos):
    nb_samples = sum([len(b) for b in cache['rois']])
    h5_end = pos + nb_samples
    permutation = np.random.permutation(nb_samples)

    def write_dataset(dataset, data):
        data = np.concatenate(data)
        assert len(data) == nb_samples, \
            "data {} != nb_samples {}".format(len(data), nb_samples)
        dataset.resize(h5_end, axis=0)
        dataset[pos:h5_end] = data[permutation]

    write_dataset(h5['tags'], cache['rois'])
    write_dataset(h5['saliencies'], cache['saliencies'])
    return h5_end


def clear_cache():
    return {'rois': [], 'saliencies': []}


def parallel_load(image_pathfile, localizer_results, nb_worker=8):
    q = queue.Queue(maxsize=4*nb_worker)
    todo = queue.Queue()
    threads = []

    for image_results in localizer_results:
        todo.put(image_results)

    def first_part(fname):
        basename = os.path.basename(fname)
        name, ext = os.path.splitext(basename)
        return name.split(".")[0]

    extract_images = {}
    with open(image_pathfile) as f:
        for line in f.readlines():
            fname = line.rstrip("\n")
            extract_images[first_part(fname)] = fname

    def worker():
        while True:
            try:
                im_json = todo.get(block=False)
                localizer_fname = os.path.basename(im_json['filename'])
                fname = extract_images[first_part(localizer_fname)]
                image = imread(fname)
            except OSError as e:
                print(e)
                q.put(None)
                continue
            except queue.Empty:
                return
            q.put((im_json, image))
            todo.task_done()

    for i in range(nb_worker):
        t = threading.Thread(target=worker)
        t.start()
        threads.append(t)

    def generator():
        for i in range(len(localizer_results)):
            res = q.get()
            if res is None:
                continue
            yield res
    return generator()


def run(json_file, hdf5_fname, roi_size, image_pathfile, nb_images, threshold, offset):
    assert not os.path.exists(hdf5_fname), \
        "hdf5 file already exists: {}".format(hdf5_fname)
    tag_positions = json.load(json_file)
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
    images = tag_positions["images"]
    if nb_images is not None:
        images = images[:nb_images]
    progbar = keras.utils.generic_utils.Progbar(len(images))
    nb_batches = 64
    for i, (im_json, image) in enumerate(parallel_load(image_pathfile, images)):
        candidates = np.array(im_json['candidates'])
        saliency = np.array(im_json['saliencies']).reshape((-1))
        assert len(candidates) == len(saliency)
        selection = saliency >= threshold
        saliency = saliency[selection]
        selected_candid = candidates[selection] + offset
        rois, mask = extract_rois(selected_candid, image, roi_shape)
        cache['rois'].append(rois)
        cache['saliencies'].append(saliency[mask])
        if len(cache['rois']) > nb_batches:
            h5_pos = write_cache_to_hdf5(cache, h5, h5_pos)
            cache = clear_cache()
        progbar.update(i+1)

    if cache['rois']:
        write_cache_to_hdf5(cache, h5, h5_pos)


def main():
    parser = argparse.ArgumentParser(
        description='Build a hdf5 dataset with the tags')
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
    parser.add_argument('-i', '--image-pathfile', type=str,
                        help='extraction will be done on this images.')
    parser.add_argument('--nb-images', type=int,
                        help='number images to process, usefull for testing.')
    parser.add_argument('input', type=argparse.FileType('r'),
                        help='json file from `find_tags.py`')
    args = parser.parse_args()
    run(args.input, args.out, args.roi_size, args.image_pathfile, args.nb_images,
         args.threshold, args.offset)

if __name__ == "__main__":
    main()

