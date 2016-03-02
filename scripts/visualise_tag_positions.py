#! /usr/bin/env python
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from localizer.util import extract_rois
from localizer.visualization import get_roi_overlay, plot_sample_images
from os.path import join
from scipy.misc import imread, imsave
import os
import json
import argparse
import numpy as np


def plot_mean_image(mean_image):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xticks(np.arange(0, 64, 8))
    ax.set_yticks(np.arange(0, 64, 8))
    plt.grid(True)
    plt.imshow(mean_image, cmap='gray')
    plt.colorbar()


def main(json_file, output_dir, nb_overlays):
    os.makedirs(output_dir, exist_ok=True)
    images = json.load(json_file)['images']
    all_rois = []
    roi_shape = (64, 64)
    saliencies = [s for im in images for s in im['saliencies']]
    for i, (fname, candidates) in enumerate(
            [(im['filename'], im['candidates']) for im in images]):
        image = imread(fname) / 255
        candidates = np.array(candidates)
        magic_offset_correction = [[5, 5]]
        offset = np.repeat(magic_offset_correction, len(candidates), axis=0)
        rois = extract_rois(candidates - offset, image, roi_shape)
        all_rois.append(rois)
        if i < nb_overlays:
            overlay = get_roi_overlay(candidates - offset, image)
            fname = join(output_dir, '{}.png'.format(i))
            print("Saving overlay image to {}".format(fname))
            imsave(fname, overlay)

    all_rois = np.concatenate(all_rois, axis=0)
    plot_mean_image(all_rois.mean(axis=0)[0])
    fname = join(output_dir, 'mean_image.png')
    print("Saving mean image to {}".format(fname))
    plt.savefig(fname)
    plt.clf()
    fig = plot_sample_images(all_rois, saliencies, random=True)
    fname = join(output_dir, 'sample_tags.png')
    print("Saving sample tags to {}".format(fname))
    fig.savefig(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='save images with roi, example tags and the rois mean'
        ' image.')
    parser.add_argument('-o', '--output-dir', type=str, default='.',
                        help='directory where to write the output.')
    parser.add_argument('input', type=argparse.FileType('r'),
                        help='json file with the tag positions.')
    parser.add_argument('--nb-overlays', type=int, default=3,
                        help='number of file to save with roi overlay.')
    arg = parser.parse_args()
    main(arg.input, arg.output_dir, arg.nb_overlays)
