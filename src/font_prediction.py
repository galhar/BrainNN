# Writer: Gal Harari
# Date: 14/12/2020
import os
import cv2 as cv
import numpy as np

from src.utils.train_utils import ClassesDataLoader


class FontDataLoader(ClassesDataLoader):

    def __init__(self, data_dir, batched=False, shuffle=False, noise_std=0):
        """
        :param noise_std: This std is multiplied by the amplitude. Noise might cause the
        model to be more robust, like dropout in ANNs. Noise can be generated during
        training for each insertion of the input instead, using NetWrapper. Maybe it's
        better.
        """
        data_array = FontDataLoader.load_font_data(data_dir)
        super().__init__(data_array,batched,shuffle,noise_std)




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

