#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import random

AUTOTUNE = tf.data.experimental.AUTOTUNE

import pathlib

data_url='https://drive.google.com/open?id=1OySoPTyWU-2_LyEN6RZjN0q5FacK4gme'
data_root_orig = tf.keras.utils.get_file(origin=data_url, fname='CheetahML_images.tar.gz', extract=True, archive_format='tar' )
data_root = pathlib.Path(data_root_orig)
print("INFO: data path:", data_root)
#for item in data_root.iterdir():
#  print(item)
#data_root = pathlib.Path("images")
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

SIZEX = 64
SIZEY = 64
NCHAN = 3


def preprocess_image(image, sizex=SIZEX, sizey=SIZEY):
    image = tf.image.decode_jpeg(image, channels=NCHAN)
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

BATCH_SIZE = 32
ds = image_label_ds.shuffle(buffer_size=image_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)

from tensorflow.keras import datasets, layers, models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                        input_shape=(SIZEX, SIZEY, NCHAN)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(3, activation='softmax'))
model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(ds, epochs=30, steps_per_epoch=5)
print("INFO: fit step done.")

test_loss, test_acc = model.evaluate(ds, steps=3)
print("INFO: test accuracy:", test_acc)
