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
    for idx, (r, c) in enumerate(coordinates):
        pltim[int(np.ceil(r - config.data_imsize[0] / 2)):int(np.ceil(r + config.data_imsize[0] / 2)),
              int(np.ceil(c - config.data_imsize[1] / 2)):int(np.ceil(c + config.data_imsize[1] / 2)),
              0] = 1.

    return pltim


def get_circle_overlay(coordinates, image, radius=32, line_width=8):
    height, width = image.shape
    overlay = np.stack([image, image, image], axis=-1)
    import cairocffi as cairo
    image_surface = cairo.ImageSurface(cairo.FORMAT_A8, image.shape[1], image.shape[0])
    ctx = cairo.Context(image_surface)
    for x, y in coordinates:
        ctx.save()
        ctx.translate(int(y), int(x))
        ctx.new_path()
        ctx.arc(0, 0, radius + line_width / 2., 0, 2*np.pi)
        ctx.close_path()
        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.set_line_width(line_width)
        ctx.stroke()
        ctx.restore()

    image_surface.flush()
    circles = np.ndarray(shape=(height, width),
                         buffer=image_surface.get_data(),
                         dtype=np.uint8)
    circles_mask = (circles == 255)
    overlay[circles_mask, 0] = 1
    overlay[circles_mask, 1] = 0
    overlay[circles_mask, 2] = 0
    image_surface.finish()
    return overlay
