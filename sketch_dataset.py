"""
Learning to simplify
Implementd by Keras

Licensed under the MIT License (see LICENSE for details)
"""

import numpy as np
import cv2
import glob
import scipy.io
import os

class Sketch_Dataset(object):

    def __init__(self, dataset_path, subset="train", shuffle=True):
        type_list = ["train", "validation", "test"]
        assert subset in type_list, "subset should in \
                chose from train, validation or test."
        self.dataset_path = dataset_path
        self.dataset_imageA_path = dataset_path + subset + "/groupa/"
        self.dataset_imageB_path = dataset_path + subset + "/groupb/"
        self.subset = type_list.index(subset) + 1
        self.shuffle = shuffle
        self.imageA_list = []
        self.imageB_list = []
        self.image_ids = []
        self._load_file_names()

    def _load_file_names(self):
        image_files_A = glob.glob(self.dataset_imageA_path + "*.jpg")
        image_files_B = glob.glob(self.dataset_imageB_path + "*.jpg")
        self.imageA_list = sorted(image_files_A)
        self.imageB_list = sorted(image_files_B)

    def load_image_gt(self, image_id):
        image_a = cv2.imread(self.imageA_list[image_id])
        image_a = cv2.bitwise_not(image_a)
        image_b = cv2.imread(self.imageB_list[image_id], 0)
        image_b = cv2.bitwise_not(image_b)
        image_b = np.expand_dims(image_b,axis=-1)

        return image_a, image_b

    def prepare(self):
        assert len(self.imageA_list) == len(self.imageB_list), \
                "number of imageA_list is not equal to imageB_list."

        self.num_images = len(self.imageA_list)
        self.image_ids = np.arange(self.num_images)

if __name__ == '__main__':
    DATASET_PATH = "./Data/"
    dataset = Sketch_Dataset(DATASET_PATH, "train")
    image_a, image_b = dataset.load_image_gt(2)

    cv2.imshow("image_A", image_a)
    cv2.imshow("image_B", image_b)
    cv2.waitKey(0)



