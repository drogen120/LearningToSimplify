import sys
import os
import math
import random
import numpy as np
import keras
import tensorflow as tf
import cv2
import scipy.misc
import warnings

def unmold_image_a(normalized_images, mean_pixel=np.array([123.0, 123.0, 123.0])):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images * 255.0).astype(np.uint8)
def unmold_image_b(normalized_images, mean_pixel=np.array([123.0])):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images * 255.0).astype(np.uint8)

def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = scipy.misc.imresize(
            image, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding

def resize_image_b(mask, scale, padding):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.
    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask = scipy.ndimage.zoom(mask, zoom=[scale, scale, 1], order=0)
    mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask

class Sketch_Evaluator(keras.callbacks.Callback):
    def __init__(self, data_gen, log_dir=None, eval_num=1):
        super().__init__()
        self.data_gen = data_gen
        self.evaluation_nums = eval_num
        self.log_dir = log_dir

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, epoch, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        self.image_front = None
        self.image_linedrawing = None
        self.image_pred = None

    def on_epoch_end(self, epoch, logs={}):
        [val_image_front, val_image_right, val_image_left, val_image_up, val_image_down], val_image_linedrawing = next(self.data_gen)
        image_pred = self.model.predict([val_image_front, val_image_right, val_image_left, val_image_up, val_image_down])
        self.image_front = unmold_image_a(val_image_front[0,:,:,:])
        self.image_linedrawing = unmold_image_b(val_image_linedrawing[0,:,:,:])
        # self.image_b = val_gt_mask[0,:,:].astype(np.float32)
        self.image_pred = unmold_image_b(image_pred[0,:,:,:])
        self.log_result_tensorboard(epoch)

    def on_batch_begin(self, batch, logs={}):
        return 

    def on_batch_end(self, batch, logs={}):
        return

    def log_result_tensorboard(self, epoch):
        if self.log_dir is not None:
            image_folder = self.log_dir
            if not os.path.exists(image_folder):
                os.mkdir(image_folder)
            
            print("image folder:", image_folder)
            cv2.imwrite(image_folder+"image_front_"+str(epoch)+".jpg",
                        self.image_front)
            cv2.imwrite(image_folder+"image_gt_"+str(epoch)+".jpg",
                        self.image_linedrawing)
            cv2.imwrite(image_folder+"image_pred_"+str(epoch)+".jpg",
                        self.image_pred)
            # logger = Logger(self.log_dir)
            # logger.log_images("Image_A", [self.image_a], epoch)
            # logger.log_images("Image_B", [self.image_b], epoch)
            # logger.log_images("Image_Pre", [self.image_pred], epoch)
