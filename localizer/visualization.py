#!/usr/bin/python3

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from localizer import config

def plot_sample_images(X, y, y_true = None, num = 24, rowsize = 6, y_bool=False, fig=None, random=False):
    if fig is None:
        fig = plt.figure(figsize=config.default_figsize)

    assert(num % rowsize == 0)

    if random:
        data_indices = np.random.choice(X.shape[0], num, replace=False)
    else:
        data_indices = range(num)
    for idx in range(num):
        ax = plt.subplot(num / rowsize, rowsize, idx + 1)
        ax.imshow(X[data_indices[idx], 0, :, :], cmap=plt.cm.gray)
        title = y[data_indices[idx]][1] == 1 if y_bool else '{0:.2f}'.format(y[data_indices[idx]][0])
        if y_true is not None:
            y_true_title = y_true[data_indices[idx]][1] == 1 if y_bool else '{0:.2f}'.format(y_true[data_indices[idx]][0])
            title = '{} | {}'.format(title, y_true_title)
        ax.set_title(title)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    plt.tight_layout()

    return fig

def plot_roc(fpr, tpr, roc_auc, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=config.default_figsize)

    ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    ax.grid()

    return ax

def plot_recall_precision(recall, precision, average_precision, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=config.default_figsize)

    ax.plot(recall, precision, label='Precision-Recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title('Precision-Recall: AUC={0:0.2f}'.format(average_precision))
    ax.legend(loc="lower left")
    ax.grid()

    return ax

def plot_saliency_image(image, saliency, filtersize, figsize=(16, 8)):
    fig, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=figsize)
    axes.flat[0].imshow(image[filtersize[0]//2:-filtersize[0]//2, filtersize[1]//2:-filtersize[1]//2], cmap=plt.cm.gray)
    axes.flat[0].axis('off')
    im = axes.flat[1].imshow(saliency, cmap=plt.cm.Blues)
    axes.flat[1].axis('off')

    plt.tight_layout()
    cax, kw = mpl.colorbar.make_axes([ax for ax in axes.flat], shrink=0.75)
    plt.colorbar(im, cax=cax, **kw)

    return fig, axes

def get_roi_overlay(coordinates, image):
    pltim = np.zeros((image.shape[0], image.shape[1], 3))
    pltim[:, :, 0] = image
    pltim[:, :, 1] = image
    pltim[:, :, 2] = image
    rois = np.zeros((len(coordinates), 1, config.data_imsize[0], config.data_imsize[1]))
    saliencies = np.zeros((len(coordinates), 1))
    scale = config.data_imsize[0] / config.filtersize[0]
    assert(config.data_imsize[0] == config.data_imsize[1])
    assert(config.filtersize[0] == config.filtersize[1])
    for idx, (r, c) in enumerate(coordinates):
        rc = r + (config.filtersize[0] - 1) / 2
        cc = c + (config.filtersize[1] - 1) / 2
        assert(int(np.ceil(rc - config.filtersize[0] / 2)) == r)
        assert(int(np.ceil(rc + config.filtersize[0] / 2)) == r + config.filtersize[0])

        rc_orig = rc * scale
        cc_orig = cc * scale
        pltim[int(np.ceil(rc_orig - config.data_imsize[0] / 2)):int(np.ceil(rc_orig + config.data_imsize[0] / 2)),
              int(np.ceil(cc_orig - config.data_imsize[1] / 2)):int(np.ceil(cc_orig + config.data_imsize[1] / 2)),
              0] = 1.

    return pltim

