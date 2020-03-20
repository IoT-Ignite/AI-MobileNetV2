from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import cv2
import numpy as np
import IPython.display as display
import sys
import argparse
import pathlib
import matplotlib.pyplot as plt
import pandas as pd
import PIL.Image
from mobilenetv2 import MobileNetV2

import os
import library.standard_fields as fields
#tf.enable_eager_execution()

slim_example_decoder = tf.contrib.slim.tfexample_decoder
image_feature_description = {
        'image/height':tf.io.FixedLenFeature([], tf.int64),
        'image/width':tf.io.FixedLenFeature([], tf.int64),
        'image/filename':tf.io.FixedLenFeature([], tf.string),
        'image/source_id':tf.io.FixedLenFeature((), tf.string),
        'image/key/sha256':tf.io.FixedLenFeature((), tf.string),
        'image/encoded': tf.io.FixedLenFeature([], tf.string),
        'image/format':tf.io.FixedLenFeature((), tf.string),
        'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':tf.io.VarLenFeature(tf.float32),
        'image/object/class/text':tf.io.VarLenFeature(tf.string),
        'image/object/class/label':tf.io.VarLenFeature(tf.int64),
        }
training_filename="./tfrecord/train.tfrecord"
validation_filename="./tfrecord/val.tfrecord"
AUTO = tf.data.experimental.AUTOTUNE

VALIDATION_SPLIT = 0.19

BATCH_SIZE = 32
HASH_KEY = 'hash'
HASH_BINS = 1 << 31
SERVING_FED_EXAMPLE_KEY = 'serialized_example'
# input layer name
items_to_handlers = {
        fields.InputDataFields.image: slim_example_decoder.Image(
        image_key='image/encoded', format_key='image/format', channels=3),
        fields.InputDataFields.source_id: (
        slim_example_decoder.Tensor('image/source_id')),
        fields.InputDataFields.key: (
        slim_example_decoder.Tensor('image/key/sha256')),
        fields.InputDataFields.filename: (
        slim_example_decoder.Tensor('image/filename')),
        # Object boxes and classes.
        fields.InputDataFields.groundtruth_boxes: (
        slim_example_decoder.BoundingBox(
        ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/')),
        fields.InputDataFields.groundtruth_classes: (
        slim_example_decoder.Tensor('image/object/class/label'))
    }
    
def imgs_input_fn(filenames, perform_shuffle=False, repeat_count=1, batch_size=1):
    def read_decode(example_proto):
        #key, serialized_example = reader.read(fname_queue)
        features = tf.io.parse_single_example(example_proto,
                                features = image_feature_description)
        
        height = features['image/height']
        width = features['image/width']
        filename=features["image/filename"]
        #image_raw = tf.image.decode_png(image_encoded, channels=3)
        #with tf.name_scope('decode_jpeg', [image_buffer], None):
        source_id=features["image/source_id"]
        key=features["image/key/sha256"]
        forma1=features['image/format']
        image_buffer = features["image/encoded"]
        
        #bbox
        xmin = tf.cast(features['image/object/bbox/xmin'], tf.float32)
        ymin = tf.cast(features['image/object/bbox/ymin'], tf.float32)
        xmax = tf.cast(features['image/object/bbox/xmax'], tf.float32)
        ymax = tf.cast(features['image/object/bbox/ymax'], tf.float32)
        class_text=features['image/object/class/text']
        label=features['image/object/class/label']
        bbox = [ymin,xmin,ymax,xmax]

        image = tf.image.decode_image(image_buffer)
        image = tf.cast(image, tf.float32)
        image_shape = tf.stack([224, 224, 3])
        #image = tf.reshape(image, image_shape)


        #label
        label = {
            "image/object/bbox/xmin" : tf.sparse.to_dense(features["image/object/bbox/xmin"], default_value=-1),
            "image/object/bbox/ymin" : tf.sparse.to_dense(features["image/object/bbox/ymin"], default_value=-1),
            "image/object/bbox/xmax" : tf.sparse.to_dense(features["image/object/bbox/xmax"], default_value=-1),
            "image/object/bbox/ymax" : tf.sparse.to_dense(features["image/object/bbox/ymax"], default_value=-1),
            "image/object/class/label" : features["image/object/class/label"],
            }
        return image, label

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(read_decode)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)#dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

def main():

    parser = argparse.ArgumentParser()
    # Required arguments.
    parser.add_argument(
        "--train_file",
        default="./tfrecord/train.tfrecord",
        help="path to train.tfrecord file")
    parser.add_argument(
        "--valid_file",
        default="./tfrecord/val.tfrecord",
        help="path to validation file")

    args = parser.parse_args()

    myModel=MobileNetV2((224, 224, 3), 1, loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    createdModel=myModel.create_model()

    print("MobileNetV2 summary")
    print(createdModel.summary())

    trained_model_dir = os.path.join(os.getcwd(), "mobV2")
    os.makedirs(trained_model_dir, exist_ok=True)
    print("trained_model_dir: ", trained_model_dir)
    input_name = createdModel.input_names[0]
    #with CustomObjectScope({'relu6': relu6}):
    #    createdModel= createdModel
    #    createdModel.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(),  metrics=metrics)
    #mobilenet_estimator = tf.keras.estimator.model_to_estimator(keras_model=keras_mobilenet)
    mobilenetModel = tf.keras.estimator.model_to_estimator(keras_model=createdModel, model_dir=trained_model_dir)



    train_spec = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(args.train_file ,perform_shuffle=True,repeat_count=5,batch_size=20), max_steps=500)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(args.valid_file, perform_shuffle=False, batch_size=1))
    tf.estimator.train_and_evaluate(mobilenetModel, train_spec, eval_spec)

if __name__ == '__main__':
    main()