# Writer: Gal Harari
# Date: 14/12/2020
import os
import cv2 as cv
import numpy as np
import cv2

from src.utils.train_utils import ClassesDataLoader

N = 10


def get_rep(i, n):
    rep = np.zeros((n,))
    rep[i] = 1
    return rep


def get_paired_perm_rep(i, n):
    rep = np.zeros((2 * n,))
    rep[(i + 3) % (2 * n)] = 1
    rep[(n + i + 3) % (2 * n)] = 1
    return rep


def get_tripled_perm_rep(i, n):
    rep = np.zeros((2 * n,))
    rep[(i + 3) % (2 * n)] = 1
    rep[(n + i + 3) % (2 * n)] = 1
    rep[(2 * (i + 3)) % (2 * n)] = 1
    return rep


class IdentityDataLoader(ClassesDataLoader):

    def __init__(self, batched=False, shuffle=True, noise_std=0, amp=30):
        """
        :param noise_std: This std is multiplied by the amplitude. Noise might cause the
        model to be more robust, like dropout in ANNs. Noise can be generated during
        training for each insertion of the input instead, using NetWrapper. Maybe it's
        better.
        """
        self.amp = amp
        data_array = [(i, amp * get_tripled_perm_rep(i, N)) for i in range(N)]
        super().__init__(data_array, batched, shuffle, noise_std)
