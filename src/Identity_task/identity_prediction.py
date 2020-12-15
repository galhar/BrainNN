# Writer: Gal Harari
# Date: 14/12/2020
import os
import cv2 as cv
import numpy as np
import cv2

from src.utils.train_utils import ClassesDataLoader

N = 100


def get_rep(i, n):
    rep = np.zeros((n,))
    rep[i] = 30
    return rep


class IdentityDataLoader(ClassesDataLoader):

    def __init__(self, batched=False, shuffle=True, noise_std=0):
        """
        :param noise_std: This std is multiplied by the amplitude. Noise might cause the
        model to be more robust, like dropout in ANNs. Noise can be generated during
        training for each insertion of the input instead, using NetWrapper. Maybe it's
        better.
        """
        data_array = [(i, get_rep(i, N)) for i in range(N)]
        super().__init__(data_array, batched, shuffle, noise_std)
