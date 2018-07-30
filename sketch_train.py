import numpy as np
import cv2
import sys
import imgaug

import os

import utils
from config import Config
from sketch_dataset import Sketch_Dataset
import sketch_network 

import keras.backend as K
import tensorflow as tf
from tensorflow.python import debug as tf_debug

ROOT_DIR = os.getcwd()
DATASET_PATH = os.path.join(ROOT_DIR, "Data/")
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

class SketchConfig(Config):
    NAME = "Sketch"
    TRAIN_BN = True
    STEPS_PER_EPOCH = 100
    VALIDATION_STEPS = 4

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train SketchNet')

    parser.add_argument('--continue_train', required=False,
                        type = bool, default = False,
                        metavar=None,
                        help="Continue Train")

    args = parser.parse_args()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    tf_sess = tf.Session(config=tf_config)
    K.set_session(tf_sess)

    config = SketchConfig()
    
    model = sketch_network.SketchNet(mode="training", config=config,
                                     model_dir=DEFAULT_LOGS_DIR)

    exclude_layers = []
    if args.continue_train == True:
        # Find last trained weights
        model_path = model.find_last()[1]
        model.load_weights(model_path, by_name=True, exclude=exclude_layers)

    dataset_train = Sketch_Dataset(DATASET_PATH, "train")
    dataset_train.prepare()
    print("dataset ids", dataset_train.image_ids)

    dataset_val = Sketch_Dataset(DATASET_PATH, "train")
    dataset_val.prepare()

    augmentation = imgaug.augmenters.Sometimes(0.5, [
    imgaug.augmenters.Fliplr(0.5),
    imgaug.augmenters.Flipud(0.5),
    imgaug.augmenters.Sometimes(0.5, imgaug.augmenters.Affine(
        scale={"x":(0.8,1.2), "y":(0.8, 1.2)},
        rotate=(-90, 90)))
    ])

    model.train(dataset_train, dataset_val, learning_rate = 0.001,
                   epochs = 400, layers = 'all', augmentation=augmentation)

