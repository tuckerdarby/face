import tensorflow as tf
import numpy as np
import os
import provider
from constants import *


def triplet_loss(anchor, positive, negative, alpha):
    with tf.variable_scope('triplet_loss'):
        positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        total_loss = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(total_loss, 0.0), 0)

    return loss


def face_trainer(model, learning_rate, image_shape, reuse=True):
    anchors = tf.placeholder(tf.float32, image_shape)
    positives = tf.placeholder(tf.float32, image_shape)
    negatives = tf.placeholder(tf.float32, image_shape)
    alpha = tf.placeholder(tf.float32, None)

    anchors_ = model(anchors, reuse=reuse)
    positives_ = model(positives, reuse=reuse)
    negatives_ = model(negatives, reuse=reuse)

    loss = triplet_loss(anchors_, positives_, negatives_, alpha)
    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    points = {
        'anchors': anchors,
        'positives': positives,
        'negatives': negatives,
        'alpha': alpha,
        'loss': loss,
        'train': trainer
    }

    return points


def face_eval(model, run_name, images, image_shape):
    inbound = tf.placeholder(tf.float32, image_shape)
    logits = model(inbound, reuse=True)
    restorer = tf.train.Saver()
    embeddings = []

    with tf.Session() as sess:
        tf.global_variables_initializer()
        checkpoint =  tf.train.get_checkpoint_state(CHECKPOINT_LOC + run_name)
        if checkpoint and checkpoint.model_checkpoint_path:
            print 'restoring model'
            restorer.restore(sess, checkpoint.model_checkpoint_path)
        else:
            print 'bad restore model'
        for i in range(len(images)):
            feed_dict = {inbound: images[i]}
            embeddings.append(sess.run(logits, feed_dict))

    return embeddings


def face_train(model, run_name, max_iter=100, people=10, samples=30, alpha=1, learning_rate=0.001, image_shape=None):
    if image_shape is None:
        images = provider.sample_people(num_people=1, samples=1)
        image_shape = (None, images[0].shape[1], images[0].shape[2], images[0].shape[3])

    inbound = tf.placeholder(tf.float32, image_shape)
    logits = model(inbound)
    trainer = face_trainer(model, learning_rate, image_shape)

    saver = tf.train.Saver()
    restorer = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        if not os.path.exists(CHECKPOINT_LOC + run_name):
            os.makedirs(CHECKPOINT_LOC + run_name)
        else:
            checkpoint = tf.train.get_checkpoint_state(CHECKPOINT_LOC + run_name)
            if checkpoint and checkpoint.model_checkpoint_path:
                print 'restoring model'
                restorer.restore(sess, checkpoint.model_checkpoint_path)

        for iteration in range(max_iter):
            images = provider.sample_people(num_people=people, samples=samples, process=True)
            embeddings = []
            for i in range(len(images)):
                feed_dict = {inbound: images[i]}
                logits_ = sess.run(logits, feed_dict)
                embeddings.append(logits_)

            embeddings = np.array(embeddings)
            triplet_idxs = provider.select_triplets(embeddings)
            triplets = provider.build_batch(images, triplet_idxs)
            train_dict = {
                trainer['anchors']: triplets[0],
                trainer['positives']: triplets[1],
                trainer['negatives']: triplets[2],
                trainer['alpha']: alpha
            }

            _, loss = sess.run([trainer['train'], trainer['loss']], feed_dict=train_dict)
            print iteration, loss

            if iteration % 25 == 0:
                saver.save(sess, CHECKPOINT_LOC + run_name + '/train.ckpt')

        saver.save(sess,  CHECKPOINT_LOC + run_name + '/train.ckpt')
