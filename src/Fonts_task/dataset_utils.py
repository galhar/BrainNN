# Writer: Gal Harari
# Date: 11/01/2021
import numpy as np
import cv2
import os
from imutils import resize
from src.utils.general_utils import get_pparent_dir
from src.Fonts_task.font_prediction import FontDataLoader, IMG_SIZE

DATA_DIR = os.path.join(get_pparent_dir(__file__), 'data/Font images/')
HEIGHT = 12


def resize_all(imgs_dir, end=-1):
    if imgs_dir[-1] != '/' and imgs_dir[-1] != '\\':
        imgs_dir += '/'
    images_list = os.listdir(imgs_dir)
    if end < 0:
        end = len(images_list)

    counter = 0
    for file_name in images_list:
        if counter >= end:
            break
        if 'resized' not in file_name:
            label = file_name.split(".bmp")[0]
            img = cv2.imread(imgs_dir + file_name, cv2.IMREAD_GRAYSCALE)
            if img.shape[0] > img.shape[1]:
                resized = resize(img, height=HEIGHT)
            else:
                resized = resize(img, width=HEIGHT)
            # resized = cv2.threshold(resized, 160, 255, cv2.THRESH_BINARY)[1]
            # cv2.imshow('threshed',thresh)
            # cv2.waitKey()
            cv2.imwrite(imgs_dir + label + '_resized12.bmp', resized)
            counter += 1


def view_data():
    data_array = FontDataLoader.load_font_data(DATA_DIR + "Calibri Font images/")
    for img in data_array:
        cv2.imshow('img ', img[1].reshape(IMG_SIZE, IMG_SIZE))
        cv2.waitKey()


def resize_data():
    for font_dir in os.listdir(DATA_DIR):
        resize_all(os.path.join(DATA_DIR, font_dir))


if __name__ == '__main__':
    # view_data()
    # resize_data()
    pass
