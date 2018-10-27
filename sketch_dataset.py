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
        self.dataset_front_path = dataset_path + subset + "/front/"
        self.dataset_right_path = dataset_path + subset + "/right/"
        self.dataset_left_path = dataset_path + subset + "/left/"
        self.dataset_up_path = dataset_path + subset + "/up/"
        self.dataset_down_path = dataset_path + subset + "/down/"
        self.dataset_linedrawing_path = dataset_path + subset + "/linedrawing/"
        self.subset = type_list.index(subset) + 1
        self.shuffle = shuffle
        self.image_front_list = []
        self.image_right_list = []
        self.image_left_list = []
        self.image_up_list = []
        self.image_down_list = []
        self.image_linedrawing_list = []
        self.image_ids = []
        self._load_file_names()

    def _load_file_names(self):
        image_files_front = glob.glob(self.dataset_front_path + "*.jpg")
        image_files_right = glob.glob(self.dataset_right_path + "*.jpg")
        image_files_left = glob.glob(self.dataset_left_path + "*.jpg")
        image_files_up = glob.glob(self.dataset_up_path + "*.jpg")
        image_files_down = glob.glob(self.dataset_down_path + "*.jpg")
        image_files_linedrawing = glob.glob(self.dataset_linedrawing_path + "*.jpg")
        self.image_front_list = sorted(image_files_front)
        self.image_right_list = sorted(image_files_right)
        self.image_left_list = sorted(image_files_left)
        self.image_up_list = sorted(image_files_up)
        self.image_down_list = sorted(image_files_down)
        self.image_linedrawing_list = sorted(image_files_linedrawing)

    def load_image_gt(self, image_id):
        image_front = cv2.imread(self.image_front_list[image_id])
        image_front = cv2.bitwise_not(image_front)
        image_right = cv2.imread(self.image_right_list[image_id])
        image_right = cv2.bitwise_not(image_right)
        image_left = cv2.imread(self.image_left_list[image_id])
        image_left = cv2.bitwise_not(image_left)
        image_up = cv2.imread(self.image_up_list[image_id])
        image_up = cv2.bitwise_not(image_up)
        image_down = cv2.imread(self.image_down_list[image_id])
        image_down = cv2.bitwise_not(image_down)
        image_linedrawing = cv2.imread(self.image_linedrawing_list[image_id], 0)
        image_linedrawing = cv2.bitwise_not(image_linedrawing)
        image_linedrawing = np.expand_dims(image_linedrawing, axis=-1)

        return image_front, image_right, image_left, image_up, image_down, image_linedrawing

    def prepare(self):
        assert len(self.imageA_list) == len(self.imageB_list), \
                "number of imageA_list is not equal to imageB_list."

        self.num_images = len(self.image_front_list)
        self.image_ids = np.arange(self.num_images)

if __name__ == '__main__':
    DATASET_PATH = "./Data/"
    dataset = Sketch_Dataset(DATASET_PATH, "train")
    image_front, image_right, image_left, image_up, image_down, \
            image_linedrawing = dataset.load_image_gt(2)
    cv2.imshow("image_front", image_front)
    cv2.imshow("image_right", image_right)
    cv2.imshow("image_left", image_left)
    cv2.imshow("image_up", image_up)
    cv2.imshow("image_down", image_down)
    cv2.imshow("image_linedrawing", image_linedrawing)
    cv2.waitKey(0)
