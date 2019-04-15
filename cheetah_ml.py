#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib

data_root = pathlib.Path("images")
print(data_root)
for item in data_root.iterdir():
    print(item)

all_image_paths = list(data_root.glob('*/*'))
all_image_paths = [str(path) for path in all_image_paths]
random.shuffle(all_image_paths)

image_count = len(all_image_paths)
print("INFO: all images found:", image_count)

label_names = sorted(
    item.name for item in data_root.glob('*/') if item.is_dir())
print("INFO: label names:", label_names)

label_to_index = dict((name, index) for index, name in enumerate(label_names))
print("INFO: label-to-index assignment:", label_to_index)

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("INFO: First 10 labels indices: ", all_image_labels[:10])


def preprocess_image(image, sizex=128, sizey=128):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [sizex, sizey])
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# read raw data
img_path = all_image_paths[0]
#preprocessed_img = load_and_preprocess_image(img_path)

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
# load imgs, associate labels
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
label_ds = tf.data.Dataset.from_tensor_slices(
    tf.cast(all_image_labels, tf.int64))
image_label_ds = tf.data.Dataset.zip((image_ds, label_ds))


ds = tf.data.Dataset.from_tensor_slices((all_image_paths, all_image_labels))


def load_and_preprocess_from_path_label(path, label):
    return load_and_preprocess_image(path), label


image_label_ds = ds.map(load_and_preprocess_from_path_label)
image_label_ds
