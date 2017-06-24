from __future__ import division

import pandas as pd
import numpy as np
import tensorflow as tf
from math import ceil
from PIL import Image
from core import embed
from models.squeezenet import inference
from constants import *


def index_dm(model, run_name, batch=50):
    df = pd.read_csv(DM_DATA_LOC + 'index.csv')
    df = df[df['cropped'] == 1]
    df.to_csv(DM_DATA_LOC + 'cropped_index.csv')
    amount = len(df)
    batches = ceil(amount / batch)
    embeddings = []

    image_shape = (None, 32, 32, 3)
    inbound = tf.placeholder(tf.float32, image_shape)
    logits = embed(model, inbound, reuse=True, training=False)
    restorer = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer()
        checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_LOC + run_name)
        if checkpoint and checkpoint.model_checkpoint_path:
            print 'restoring model'
            restorer.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print 'bad restore model'

        for i in range(int(batches)):
            print 'batch', i
            start = i*batch
            end = (i+1)*batch
            rows = df.ix[start:end]
            images = get_images(rows)
            for k in range(len(images)):
                feed_dict = {inbound: images}
                embeddings.append(sess.run(logits, feed_dict))

    emb_df = pd.DataFrame(embeddings)
    emb_df.to_csv(DM_DATA_LOC + 'cropped_embeddings.csv')


def get_images(rows, ext='.jpg'):
    images = []
    for row in rows.iterrows():
        identity = row[1]['identity']
        filename = DM_PEOPLE_LOC + identity + '/faces/' + identity + ext
        img = Image.open(filename)
        img.load()
        img_arr = np.asarray(img, dtype=np.uint8)
        images.append(img_arr)
    return images


index_dm(inference, 'squeeze_05_06_r2')