#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np

from localizer import config

def plot_sample_images(X, y, num = 24, rowsize = 6, fig=None):
    if fig is None:
        fig = plt.figure(figsize=config.default_figsize)

    assert(num % rowsize == 0)

    data_indices = np.random.choice(X.shape[0], num)
    for idx in range(num):
        ax = plt.subplot(num / rowsize, rowsize, idx + 1)
        ax.imshow(X[data_indices[idx], 0, :, :], cmap=plt.cm.gray)
        ax.set_title(y[data_indices[idx]][1] == 1)
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

