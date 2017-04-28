import os
import tensorflow as tf
import numpy as np
from constants import *


def get_files(directory):
    if directory[len(directory)-1] != '/':
        directory += '/'
    search = directory + '*'
    filename_matcher = tf.train.match_filenames_once(search)
    filenames = os.listdir(directory)
    filename_queue = tf.train.string_input_producer(filename_matcher)

    image_reader = tf.WholeFileReader()

    image_names, image_files = image_reader.read(filename_queue)

    return filenames, image_files, image_names


def read_data(directory):

    filenames, image_files, image_names = get_files(directory)
    images = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(len(filenames)):
            image = tf.image.decode_jpeg(image_files)
            images.append(sess.run(image))

        coord.request_stop()
        coord.join(threads)

    return np.array(images)


def read_and_resize_data(directory, size):

    filenames, image_files, image_names = get_files(directory)
    images = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(len(filenames)):
            try:
                image = tf.image.decode_image(image_files, channels=3)
                image = tf.image.resize_nearest_neighbor([image], size)
                images.append(sess.run(image))
            except Exception:
                print 'invalid image -', sess.run(image_names)

        coord.request_stop()
        coord.join(threads)

    images = np.asarray(images, dtype=np.float32)
    shape = images.shape
    images = images.reshape(shape[0], shape[2], shape[3], shape[4])
    return images


def extract_data(filename='cameron_brigh'):
    inbounds = []
    if not filename.endswith('.tfrecords'):
        filename += '.tfrecords'
    filename = RECORD_LOC + filename
    # print filename
    record_iterator = tf.python_io.tf_record_iterator(filename)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        batch = int(example.features.feature['amount'].int64_list.value[0])
        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        depth = int(example.features.feature['depth'].int64_list.value[0])
        inbound = example.features.feature['inbound'].float_list.value

        remade = np.array(inbound, dtype=np.float32)
        remade = remade.reshape((height, width, depth))
        inbounds.append(remade)

    # print 'Decoded Shape', (batch, height, width, depth)
    return np.array(inbounds)
