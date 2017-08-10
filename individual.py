from __future__ import division

import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import dlib

from math import ceil
from PIL import Image
from core import embed
from constants import *
from scipy.misc import imresize

from models.squeezenet import inference
from processor import whiten


def resize_face(face):
    # return cv2.resize(face, (IMG_SHAPE[0], IMG_SHAPE[1], 1))
    return imresize(face, (IMG_SHAPE[0], IMG_SHAPE[1], 1))


def normalize_faces(faces):
    faces = map(resize_face, faces)
    return np.array(faces)


def get_points(image_target):
    if image_target.shape[2] != 3:
        print 'k'
    gray = cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY)
    predictor = dlib.shape_predictor(SHAPE_PREDICTOR)
    detector = dlib.get_frontal_face_detector()
    rects = detector(gray, 1)

    # print 'face found'
    face = image_target[rects[0].top() + 1:rects[0].bottom() - 1, rects[0].left():rects[0].right() - 1]
    return face


filename = '/home/tuck/Downloads/3092.jpg'
run_name = 'squeeze_05_06_r2'

img = Image.open(filename)
img.load()

img_arr = np.asarray(img, dtype=np.uint8)

imgs = [img_arr]

imgs = normalize_faces(imgs)

imgs = whiten(imgs)

model = inference

image_shape = (None, 32, 32, 3)
inbound = tf.placeholder(tf.float32, image_shape)
logits = embed(model, inbound, reuse=True, training=False)
restorer = tf.train.Saver()

embeddings = []

with tf.Session() as sess:
    tf.global_variables_initializer()
    checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_LOC + run_name)
    if checkpoint and checkpoint.model_checkpoint_path:
        print 'restoring model'
        restorer.restore(sess, checkpoint.model_checkpoint_path)
    else:
        print 'bad restore model'

    images = imgs
    feed_dict = {inbound: images}
    embeddings.extend(sess.run(logits, feed_dict))


emb_df = pd.DataFrame(embeddings)
emb_df.to_csv('/home/ubuntu/' + 'individual_embedding.csv')


