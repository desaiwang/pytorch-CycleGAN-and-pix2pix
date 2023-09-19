import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display

p2p_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(p2p_folder)

train_folder = os.path.join(p2p_folder, "datasets",
                            "path", "to", "data", "images_combined_train")
img_lst = os.listdir(train_folder)


# print(os.path.join(train_folder, img_lst[0]))
sample_image = tf.io.read_file(os.path.join(train_folder, img_lst[0]))
sample_image = tf.io.decode_jpeg(sample_image)
# print(sample_image.shape)

# debug
# plt.figure()
# plt.imshow(sample_image)
# plt.show()


def load(image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with rgb image
    # - one with thermal image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image


# PLOTTING SAMPLE OF IMAGE
inp, re = load(os.path.join(train_folder, img_lst[1]))
# Casting to int for matplotlib to display the images
# plt.figure()
# plt.imshow(inp / 255.0)

# plt.figure()
# plt.imshow(re / 255.0)
# plt.show()

# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 512x640 in size
IMG_WIDTH = 640
IMG_HEIGHT = 512


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


@tf.function()
def random_jitter(input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = resize(input_image, real_image, 563, 704)

    # Random cropping back to 256x256
    input_image, real_image = random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
        # Random mirroring
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

# VISUALIZE JITTERING & MIRRORING
# plt.figure(figsize=(6, 6))
# for i in range(4):
#     rj_inp, rj_re = random_jitter(inp, re)
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(rj_inp / 255.0)
#     plt.axis('off')
# plt.show()
