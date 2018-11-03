import numpy as np
import cv2
import sys
import imgaug

import argparse
import os

import utils
from config import Config
from sketch_dataset import Sketch_Dataset
import sketch_network 

import keras.backend as K
import tensorflow as tf
import sketch_network

ROOT_DIR = os.getcwd()
# modify the trained model folder name
MODEL_DIR = os.path.join(ROOT_DIR, "logs/sketch20180730T2341/")
# input the trained model name
SKETCH_MODEL_PATH = os.path.join(MODEL_DIR, "xxxxx.h5")

IMAGE_DIR = os.path.join(ROOT_DIR, "Data/test/")
RESULT_DIR = os.path.join(ROOT_DIR, "result_image/")

class InferenceConfig(Config):
    NAME = "Sketch"
    TRAIN_BN = False
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def crop_box(image_front, image_right, image_left, image_up, image_down, crop_height, crop_width):

    if image_front.shape[0] < crop_width:
        return None, None

    x = random.randint(0, image_front.shape[1]-crop_width)
    y = random.randint(0, image_front.shape[0]-crop_height)
    random_box = [y, x, y+crop_height, x+crop_width]

    return [image_front[y:y+crop_height, x:x+crop_width, :].copy(), \
           image_right[y:y+crop_height, x:x+crop_width, :].copy(), \
           image_left[y:y+crop_height, x:x+crop_width, :].copy(), \
           image_up[y:y+crop_height, x:x+crop_width, :].copy(), \
           image_down[y:y+crop_height, x:x+crop_width, :].copy()]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Test Sketch'
    )

    parser.add_argument('--mode', required=False)
    args = parser.parse_args()
    config = InferenceConfig()
    config.display()

    model = sketch_network.SketchNet(mode="inference", config=config,
                            model_dir=SKETCH_MODEL_PATH)

    model.load_weights(SKETCH_MODEL_PATH, by_name=True)

    image_files_front = glob.glob(TEST_IMAGE_DATASET_PATH + "front/" + "*.jpg")
    image_files_right = glob.glob(TEST_IMAGE_DATASET_PATH + "right/" + "*.jpg")
    image_files_left = glob.glob(TEST_IMAGE_DATASET_PATH + "left/" + "*.jpg")
    image_files_up = glob.glob(TEST_IMAGE_DATASET_PATH + "up/" + "*.jpg")
    image_files_down = glob.glob(TEST_IMAGE_DATASET_PATH + "down/" + "*.jpg")
    image_files_front = sorted(image_files_front)
    image_files_right = sorted(image_files_right)
    image_files_left = sorted(image_files_left)
    image_files_up = sorted(image_files_up)
    image_files_down = sorted(image_files_down)

    for (front_file, right_file, left_file, up_file, down_file) in zip(
    	     image_files_front, image_files_right, image_files_left, image_files_up, image_files_down):
        image_front = cv2.imread(front_file)
        image_front = cv2.bitwise_not(image_front)
        image_right = cv2.imread(right_file)
        image_right = cv2.bitwise_not(image_right)
        image_left = cv2.imread(left_file)
        image_left = cv2.bitwise_not(image_left)
        image_up = cv2.imread(up_file)
        image_up = cv2.bitwise_not(image_up)
        image_down = cv2.imread(down_file)
        image_down = cv2.bitwise_not(image_down)

        inputs_image = crop_box(image_front, image_right, image_left, image_up, image_down, 320, 320)
        result_image = model.predict(inputs_image)
        print(result_image.shape)
        cv2.imwrite(RESULT_DIR + file_name.split('/')[-1], result_image)