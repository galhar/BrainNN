# Writer: Gal Harari
# Date: 14/12/2020
import os
import cv2 as cv
import numpy as np

from src.utils.train_utils import DataLoaderBase


class FontDataLoader(DataLoaderBase):

    def __init__(self, data_dir, batched=False, shuffle=False, noise_std=0):
        """
        :param noise_std: This std is multiplied by the amplitude. Noise might cause the
        model to be more robust, like dropout in ANNs. Noise can be generated during
        training for each insertion of the input instead, using NetWrapper. Maybe it's
        better.
        """
        self.n_std = noise_std
        self._batched = batched
        self._shuffle = shuffle
        self._stopped_iter = True

        data_array = FontDataLoader.load_font_data(data_dir)
        self.classes = [l for l, img in data_array]
        self.images = [img for l, img in data_array]

        self._cur_label = self.classes[0]



    @staticmethod
    def load_font_data(data_dir):
        """
        loading the "resized" pictures into an array of shape [(<label_string>,
        <normalized_np_img>),...]
        :param data_dir:
        :return: array of the loaded data
        """

        if data_dir[-1] != '/' and data_dir[-1] != '\\':
            data_dir += '/'

        data_array = []
        for file_name in os.listdir(data_dir):
            if "resized" in file_name:
                label = file_name.split("resized")[0]
                img = cv.imread(data_dir + file_name) / 255.
                data_array.append((label, img))
        return data_array


    def __next__(self):
        # create a single batch and raise stop iteration. If last time didn't raised
        # stopIteration than this one should do it
        if not self._stopped_iter:
            self._stopped_iter = True
            raise StopIteration
        # This "next" doesn't raise stopIteration
        self._stopped_iter = False

        if self._shuffle:
            p = np.random.permutation(len(self.classes))
            return [self.classes[i] for i in p], [self.images[i] for i in p]

        return self.classes, self.images
