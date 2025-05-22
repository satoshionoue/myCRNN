#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import torch
import torch.nn as nn
from pathlib import Path

DIR_MYSELF = Path(__file__).resolve().parent
DIR_MODULE = DIR_MYSELF.parent
sys.path.append(str(DIR_MODULE))

from activation import generate_activation


def generate_normalizer(name):
    """
    Select a normalization layer used by the DeepSalience

    Args:
        name (str): the name of normalization function

    Returns:
        a normalization layer based on torch.nn.module
    """
    return dict(
        batchnorm    = nn.BatchNorm2d,
        instancenorm = nn.InstanceNorm2d,
    )[name]


class DeepSalience(nn.Module):
    """
    DeepSalience

    PDF:    https://brianmcfee.net/papers/ismir2017_salience.pdf
    GitHub: https://github.com/rabitt/ismir2017-deepsalience

    Note:
        B: batch size
        H: harmonics    (= channel)
        F: n_freqbins   (= height)
        T: n_timeframes (= width)
    """
    def __init__(self, n_channels, squeeze=True, normalizer="batchnorm", activation="relu", **_):
        super().__init__()

        # Default parameters of Batch Normalization in Keras
        chi, cho = n_channels, 1
        ch1, ch2, ch3, ch4, ch5 =  [128, 64, 64, 64, 8]

        self.norm_args = dict(
            batchnorm = dict(eps=1e-3, momentum=0.01, affine=True),
            instancenorm = dict(affine=True),
        )[normalizer]

        self.cnn = nn.Sequential(
            # layer set 1
            generate_normalizer(normalizer)(chi, **self.norm_args),
            nn.Conv2d(chi, ch1, (5, 5), stride=1, padding=2),
            generate_activation(activation),
            # layer set 2
            generate_normalizer(normalizer)(ch1, **self.norm_args),
            nn.Conv2d(ch1, ch2, (5, 5), stride=1, padding=2),
            generate_activation(activation),
            # layer set 3
            generate_normalizer(normalizer)(ch2, **self.norm_args),
            nn.Conv2d(ch2, ch3, (3, 3), stride=1, padding=1),
            generate_activation(activation),
            # layer set 4
            generate_normalizer(normalizer)(ch3, **self.norm_args),
            nn.Conv2d(ch3, ch4, (3, 3), stride=1, padding=1),
            generate_activation(activation),
            # layer set 5
            generate_normalizer(normalizer)(ch4, **self.norm_args),
            nn.ZeroPad2d((1, 1, 35, 34)),
            nn.Conv2d(ch4, ch5, (70, 3), stride=1, padding=0),
            generate_activation(activation),
            # layer set 6
            generate_normalizer(normalizer)(ch5, **self.norm_args),
            nn.Conv2d(ch5, cho, (1, 1), stride=1, padding=0),
        )

        self.cr = nn.BCEWithLogitsLoss(reduction="mean")
        self.squeeze = squeeze

    def forward(self, x):
        """
        Forward calculation to obtain logits.

        Args:
            x (torch.FloatTensor(shape=[B, H, F, T])): input images

        Returns:
            logits (torch.FloatTensor(shape=[B, 1, F, T] or [B, F, T])):
                logits obtained by the DeepSalience.
                logits.shape = [B, F, T]      (self.squeeze = True)
                             | [B, 1, F, T]   (otherwise)
        """
        x = self.cnn(x)

        if self.squeeze:
            x = x.squeeze(-3)

        return x

    def salience_map(self, x):
        """
        Calculate a saliency map

        Args:
            x (torch.FloatTensor(shape=[B, H, F, T])): input images

        Returns:
            salience_map (torch.FloatTensor(shape=[B, F, T])): probabilities
        """

        return torch.sigmoid(self.forward(x))

    def binary_mask(self, x, thres=0.5):
        """
        Calculate a binary mask

        Args:
            x (torch.FloatTensor(shape=[B, H, F, T])): input images

        Returns:
            salience_map (torch.FloatTensor(shape=[B, F, T])): binary matrices
        """
        return (self.salience_map(x) >= thres).to(torch.int64)

    def loss_fun(self, y, t):
        """
        Args:
            y (torch.FloatTensor(shape=[B, H, W])):
                the logits obtained by the forward calculation
            t (torch.FloatTensor(shape=[B, H, W])):
                the ground-truth binary masks
        Return:
            loss (float): divided by the batchsize(B) to take an average.
        """
        return self.cr(y, t)
