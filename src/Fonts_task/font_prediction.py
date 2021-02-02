# Writer: Gal Harari
# Date: 14/12/2020
import os
import cv2
import numpy as np

from src.utils.train_utils import ClassesDataLoader

IMG_SIZE = 12


def flatten_to_image(flat_img):
    return flat_img.reshape((IMG_SIZE, IMG_SIZE))


class FontDataLoader(ClassesDataLoader):
    DEFAULT = 20
    SMALL = 13
    SMALL_SHARP = 12
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


    def explore_images(self):
        for i, flatten in enumerate(self.samples):
            l = self.neuron_to_class_dict[self.classes_neurons[i]]
            cv2.imshow('image', flatten_to_image(flatten))
            print("label: %s"%(l))
            cv2.waitKey()


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
                img = cv2.imread(data_dir + file_name, 0) / 255.
                pad_image = np.ones((IMG_SIZE, IMG_SIZE))
                w, h = img.shape
                l_pad, top_pad = int((IMG_SIZE - w) / 2), int((IMG_SIZE - h) / 2)
                r_pad, bottom_pad = l_pad + w, top_pad + h
                pad_image[l_pad:r_pad, top_pad:bottom_pad] = img
                # Turn black into the high values, and white to the low value,
                # and increase the signal
                pad_image = 10 * (1 - pad_image)

                data_array.append((label, pad_image.flatten()))
        return data_array
