# Writer: Gal Harari
# Date: 14/12/2020
import os
import cv2 as cv
import numpy as np

from src.utils.train_utils import ClassesDataLoader

IMG_SIZE = 20


def flatten_to_image(flat_img):
    return np.reshape(flat_img, )


class FontDataLoader(ClassesDataLoader):
    DEFAULT = 20
    SMALL = 12
    SMALL_SHARP = 13
    _divider_dict = {DEFAULT: ' resize',
                     SMALL: '_resized12',
                     SMALL_SHARP: '_resized_sharp_12'}


    def __init__(self, data_dir, batched=False, shuffle=True, noise_std=0):
        """
        :param noise_std: This std is multiplied by the amplitude. Noise might cause the
        model to be more robust, like dropout in ANNs. Noise can be generated during
        training for each insertion of the input instead, using NetWrapper. Maybe it's
        better.
        """
        data_array = FontDataLoader.load_font_data(data_dir)
        super().__init__(data_array, batched, shuffle, noise_std)


    @staticmethod
    def load_font_data(data_dir):
        """
        loading the "resized" pictures into an array of shape [(<label_string>,
        <normalized_np_img>),...]
        :param data_dir:
        :param type:
        :return: array of the loaded data
        """

        if data_dir[-1] != '/' and data_dir[-1] != '\\':
            data_dir += '/'

        divider = FontDataLoader._divider_dict[IMG_SIZE]

        data_array = []
        for file_name in os.listdir(data_dir):
            if divider in file_name:
                label = file_name.split(divider)[0]
                # Using 0 to read image in grayscale mode
                img = cv.imread(data_dir + file_name, 0) / 255.
                pad_image = np.ones((IMG_SIZE, IMG_SIZE))
                w, h = img.shape
                l_pad, top_pad = int((IMG_SIZE - w) / 2), int((IMG_SIZE - h) / 2)
                r_pad, bottom_pad = l_pad + w, top_pad + h
                pad_image[l_pad:r_pad, top_pad:bottom_pad] = img
                # Turn black into the high values, and white to the low value,
                # and increase the signal
                pad_image = 10*(1 - pad_image)

                data_array.append((label, pad_image.flatten()))
        return data_array
