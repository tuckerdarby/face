import tensorflow as tf
import numpy as np
import cv2
import os
import aid
from PIL import Image
from constants import *


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def open_image(location, size):
    image = Image.open(location)
    image.thumbnail(size)
    image.load()
    data = np.asarray(image, dtype=np.uint8)
    return data


def open_image_cv2(location):
    img = cv2.imread(location)
    return img


def preview(img_arr):
    arr = np.array(img_arr)
    if arr.min() < 0:
        arr = arr + abs(arr.min())
    arr = arr / arr.max()
    arr *= 255
    return np.array(arr, dtype=np.uint8)


def get_people_names():
    names = os.listdir(PEOPLE_LOC)
    return names


def get_image_files(name, size=(5, 5)):
    image_names = os.listdir(PEOPLE_LOC + name)
    images_location = PEOPLE_LOC + name + '/'
    images = []
    for image_name in image_names:
        image = open_image(images_location + image_name, size)
        images.append(image)
        print image.shape
    return images


def record(inbound, name):
    """Converts a dataset to tfrecords."""
    amount = inbound.shape[0]
    rows = inbound.shape[1]
    cols = inbound.shape[2]
    depth = inbound.shape[3]
    filename = RECORD_LOC + name + '.tfrecords'

    print('Writing', filename)
    print('Shape', rows, cols, depth)
    writer = tf.python_io.TFRecordWriter(filename)
    for i in range(amount):
        flat_inbound = np.array(inbound[i].flat)
        example = tf.train.Example(features=tf.train.Features(feature={
            'amount': _int64_feature(amount),
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'inbound': _float_feature(flat_inbound)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def filter_images(images):
    filtered = []
    for image in images:
        if image.std() < 10 or image.mean() > 240 or image.mean() < 15:
            continue
        filtered.append(image)
    return filtered


def record_face(name, size, min_faces):
    imgs = aid.read_and_resize_data(PEOPLE_LOC + name, size)
    imgs = filter_images(imgs)
    record(imgs, name)
    return imgs


def record_faces(min_faces=50):
    names = get_people_names()
    size = FACE_SIZE
    for name in names:
        if os.path.exists(RECORD_LOC + name + '.tfrecords') or len(aid.get_files(PEOPLE_LOC + name)[0]) < min_faces:
            continue
        imgs = record_face(name, size, min_faces)
        if imgs is not None:
            print name, imgs.shape


record_faces()
