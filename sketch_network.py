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
import datetime
import re
import multiprocessing
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
from keras import metrics

import utils
import config
from utils import Sketch_Evaluator

def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}".format(
            str(array.shape),
            array.min() if array.size else "",
            array.max() if array.size else ""))
    print(text)

class BatchNorm(KL.BatchNormalization):
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=training)

def mold_image_a(images, config):
    return images.astype(np.float32) / 255.0
def mold_image_b(images, config):
    return images.astype(np.float32) / 255.0

def mold_images(images):
    result_images = []
    for image in images:
        result_images.append(image.astype(np.float32) / 255.0)

    return result_images

def unmold_image_a(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images * 255.0).astype(np.uint8)
def unmold_image_b(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images * 255.0).astype(np.uint8)

def random_crop(image_A, image_B ,image_C ,image_D, image_E, image_F, crop_height, crop_width):
    if (crop_width <= image_A.shape[1]) and (crop_height <= image_A.shape[0]):
        x = np.random.randint(0, image_A.shape[1]-crop_width)
        y = np.random.randint(0, image_A.shape[0]-crop_height)
        return image_A[y:y+crop_height, x:x+crop_width, :], image_B[y:y+crop_height, x:x+crop_width, :], \
               image_C[y:y+crop_height, x:x+crop_width, :], image_D[y:y+crop_height, x:x+crop_width, :], \
               image_E[y:y+crop_height, x:x+crop_width, :], image_F[y:y+crop_height, x:x+crop_width, :]

    else:
        raise Exception('Crop shape exceeds image dimensions!')

def load_image_AB(dataset, config, image_id, augmentation=None):
    image_front, image_right, image_left, image_up, image_down, image_linedrawing = dataset.load_image_gt(image_id)
    image_front, window, scale, padding = utils.resize_image(
        image_front,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)

    image_right = utils.resize_image_b(image_right, scale, padding)
    image_left = utils.resize_image_b(image_left, scale, padding)
    image_up = utils.resize_image_b(image_up, scale, padding)
    image_down = utils.resize_image_b(image_down, scale, padding)
    image_linedrawing = utils.resize_image_b(image_linedrawing, scale, padding)

    if augmentation:
        import imgaug
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

        # Store shapes before augmentation to compare
        image_front_shape = image_front.shape
        image_right_shape = image_right.shape
        image_left_shape = image_left.shape
        image_up_shape = image_up.shape
        image_down_shape = image_down.shape
        image_linedrawing_shape = image_linedrawing.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image_front = det.augment_image(image_front.astype(np.uint8))
        image_right = det.augment_image(image_right.astype(np.uint8))
        image_left = det.augment_image(image_left.astype(np.uint8))
        image_up = det.augment_image(image_up.astype(np.uint8))
        image_down= det.augment_image(image_down.astype(np.uint8))
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        image_linedrawing = det.augment_image(image_linedrawing.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image_front_shape == image_front.shape, "Augmentation shouldn't change image size"
        assert image_right_shape ==  image_right.shape, "Augmentation shouldn't change image size"
        assert image_left_shape ==  image_left.shape, "Augmentation shouldn't change image size"
        assert image_up_shape ==  image_up.shape, "Augmentation shouldn't change image size"
        assert image_down_shape ==  image_down.shape, "Augmentation shouldn't change image size"
        assert image_linedrawing_shape ==  image_linedrawing.shape, "Augmentation shouldn't change image size"

    return image_front, image_right, image_left, image_up, image_down, image_linedrawing 

def data_generator(dataset, config, shuffle=True, augmentation=None, batch_size=1):
    if dataset is None:
        return None

    b = 0
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    while True:
        try:
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            image_id = image_ids[image_index]
            image_front, image_right, image_left, image_up, image_down, image_linedrawing \
                = load_image_AB(dataset, config, image_id, augmentation=augmentation)

            image_front, image_right, image_left, image_up, image_down, image_linedrawing \
                = random_crop(image_front, image_right, image_left, image_up, image_down, image_linedrawing, 
                                           config.IMAGE_MIN_DIM, config.IMAGE_MIN_DIM)

            if b == 0:
                batch_images_front = np.zeros(
                    (batch_size,) + image_front.shape, dtype=np.float32)

                batch_images_right = np.zeros(
                    (batch_size,) + image_right.shape, dtype=np.float32)
                batch_images_left = np.zeros(
                    (batch_size,) + image_left.shape, dtype=np.float32)
                batch_images_up = np.zeros(
                    (batch_size,) + image_up.shape, dtype=np.float32)
                batch_images_down = np.zeros(
                    (batch_size,) + image_down.shape, dtype=np.float32)
                batch_images_linedrawing = np.zeros(
                    (batch_size,) + image_linedrawing.shape, dtype=np.float32)

            batch_images_front[b] = mold_image_a(image_front.astype(np.float32), config)
            batch_images_right[b] = mold_image_a(image_right.astype(np.float32), config)
            batch_images_left[b] = mold_image_a(image_left.astype(np.float32), config)
            batch_images_up[b] = mold_image_a(image_up.astype(np.float32), config)
            batch_images_down[b] = mold_image_a(image_down.astype(np.float32), config)
            batch_images_linedrawing[b] = mold_image_b(image_linedrawing.astype(np.float32), config)

            b += 1

            if b >= batch_size:
                inputs = [batch_images_front, batch_images_right, batch_images_left, batch_images_up, batch_images_down]
                outputs = batch_images_linedrawing

                yield inputs, outputs

                # start a new batch
                b = 0

        except(GeneratorExit, KeyboardInterrupt):
            raise
        except:
            error_count += 1
            print("error_count", error_count)
            if error_count > 5:
                raise

def build_sketchnet_graph(input_image_front, input_image_right, input_image_left,
                          input_image_up, input_image_down,
                          feature_network_parameter, network_parameter, use_bias = True, train_bn=True):

    name_list = ["front", "right", "left", "up", "down"]
    input_list = [input_image_front, input_image_right, input_image_left,
                          input_image_up, input_image_down]
    feature_list = []
    for (input_tensor, input_name) in zip(input_list, name_list):
        feature_output = KL.Conv2D(feature_network_parameter[0][2], 
                      (feature_network_parameter[0][0],feature_network_parameter[0][0]),
                      strides=(feature_network_parameter[0][1],feature_network_parameter[0][1]),
                      padding='same', use_bias=use_bias,
                      name='Sketch_conv_input_' + input_name)(input_tensor)
        feature_output = BatchNorm(name='Sketch_bn_conv_input_' + input_name)(feature_output, training=train_bn)
        feature_output = KL.Activation('relu')(feature_output)
        for index, conv_layer_para in enumerate(feature_network_parameter[1:]):
            kernel_size, stride_step, filter_num = conv_layer_para
            feature_output = KL.Conv2D(filter_num, (kernel_size,kernel_size),
                          strides=(stride_step,stride_step),
                          padding='same', use_bias=use_bias,
                          name='Sketch_conv_'+ input_name + str(index))(feature_output)
            feature_output = BatchNorm(name='Sketch_bn_' + input_name +
                                       str(index))(feature_output, training=train_bn)
            feature_output = KL.Activation('relu')(feature_output)

        feature_list.append(feature_output)

    x = KL.Concatenate()(feature_list)

    for index, conv_layer_para in enumerate(network_parameter[:-1]):
        kernel_size, stride_step, filter_num = conv_layer_para

        if stride_step >= 1:
            x = KL.Conv2D(filter_num, (kernel_size,kernel_size),
                          strides=(stride_step,stride_step),
                          padding='same', use_bias=use_bias, name='Sketch_conv_' + str(index))(x)
        else:
            x = KL.Conv2DTranspose(filter_num, (kernel_size, kernel_size), 
                                   strides=(int(1/stride_step),int(1/stride_step)), 
                                   padding='same', use_bias=use_bias,
                                   name='Sketch_deconv_' + str(index))(x)

        x = BatchNorm(name='Sketch_bn_' + str(index))(x, training=train_bn)
        x = KL.Activation('relu')(x)

    kernel_size, stride_step, filter_num = network_parameter[-1]
    output_image = KL.Conv2D(filter_num, (kernel_size,kernel_size),
                  strides=(stride_step,stride_step),
                  padding='same', use_bias=use_bias, name='Sketch_output')(x)

    return output_image

class SketchNet():
    def __init__(self, mode, config, model_dir):
        assert mode in ['training', 'inference']

        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()

        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        assert mode in ['training', 'inference']

        input_image_front = KL.Input(
            shape = config.IMAGE_SHAPE.tolist(), name="input_image_front"
        )
        input_image_right = KL.Input(
            shape = config.IMAGE_SHAPE.tolist(), name="input_image_right"
        )
        input_image_left = KL.Input(
            shape = config.IMAGE_SHAPE.tolist(), name="input_image_left"
        )
        input_image_up = KL.Input(
            shape = config.IMAGE_SHAPE.tolist(), name="input_image_up"
        )
        input_image_down = KL.Input(
            shape = config.IMAGE_SHAPE.tolist(), name="input_image_down"
        )
        if mode == 'training':
            output_image = build_sketchnet_graph(input_image_front,
                                                 input_image_right,
                                                 input_image_left,
                                                 input_image_up,
                                                 input_image_down,
                                                 config.NETWORK_PARM_FEATURE,
                                                 config.NETWORK_PARM,
                                                 config.TRAIN_BN)
            inputs = [input_image_front, input_image_right, input_image_left,
                      input_image_up, input_image_down]
            outputs = output_image

            model = KM.Model(inputs, outputs, name='SketchNet')
            model.summary()
        else:
            output_image = build_sketchnet_graph(input_image_front,
                                                 input_image_right,
                                                 input_image_left,
                                                 input_image_up,
                                                 input_image_down,
                                                 config.NETWORK_PARM_FEATURE,
                                                 config.NETWORK_PARM,
                                                 config.TRAIN_BN)
            inputs = [input_image_front, input_image_right, input_image_left,
                      input_image_up, input_image_down]
            outputs = output_image

            model = KM.Model(inputs, outputs, name='SketchNet')

        return model

    def compile(self, learning_rate, momentum):
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        reg_loss = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) /
                    tf.cast(tf.size(w), tf.float32)
                    for w in self.keras_model.trainable_weights
                    if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_loss))
        self.keras_model.compile(optimizer=optimizer,
                                 loss = 'mean_squared_error',
                                 metrics=['mae', 'acc'])

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("SketchNet"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/SketchNet\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "SketchNet_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers,
              augmentation=None):

        layer_regex = {
            "all" : ".*",
        }

        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        train_generator = data_generator(train_dataset, self.config,
                                         shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config,
                                         shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        vis_generator = data_generator(val_dataset, self.config,
                                         shuffle=True,
                                         augmentation=augmentation,
                                         batch_size=self.config.BATCH_SIZE)
        vis_training_image = Sketch_Evaluator(vis_generator,
                                             self.log_dir+"/visual/")

        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True,
                                        write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                           period=10,
                                           verbose=0, save_weights_only=True),
            vis_training_image,
        ]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=False,
            verbose=1,
        )
        self.epoch = max(self.epoch, epochs)

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def predict(self, images, verbose=0):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of images
        """
        assert self.mode == "inference", "Create model in inference mode."
        assert len(
            images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)
        # Mold inputs to format expected by the neural network
        molded_images = self.mold_images(images)
        if verbose:
            log("molded_images", molded_images)
        # Run predict
        result_images = self.keras_model.predict([molded_images], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(result_images):

            results.append(self.unmold_image_b(image))
        return results

if __name__ == '__main__':
    print("Sketch_network")

