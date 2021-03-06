import tensorflow as tf
import numpy as np
import os
import provider
from tensorflow.contrib import slim
from constants import *



def triplet_loss(anchor, positive, negative, alpha):
    with tf.variable_scope('triplet_loss'):
        positive_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        negative_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        total_loss = tf.add(tf.subtract(positive_dist, negative_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(total_loss, 0.0), 0)

    return loss


def embed(model, inputs, reuse, training=True, dropout=0.0):
    keep = 1 - dropout
    net, _ = model(inputs, reuse=reuse, training=training, keep_probability=keep)
    embeddings = slim.fully_connected(net, 128, activation_fn=None,
                                      weights_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                      weights_regularizer=slim.l2_regularizer(0.01),
                                      normalizer_fn=slim.batch_norm,
                                      normalizer_params={
                                          'decay': 0.995,
                                          'epsilon': 0.001,
                                          'updates_collections': None,
                                          'variables_collections': [tf.GraphKeys.TRAINABLE_VARIABLES],
                                          'is_training': training
                                      },
                                      scope='fc',
                                      reuse=reuse)
    embeddings = tf.nn.l2_normalize(embeddings, 1, 1e-10, name='embeddings')
    return embeddings


def face_trainer(model, learning_rate, image_shape, global_step, reuse=True):
    anchors = tf.placeholder(tf.float32, image_shape)
    positives = tf.placeholder(tf.float32, image_shape)
    negatives = tf.placeholder(tf.float32, image_shape)
    alpha = tf.placeholder(tf.float32, None)

    anchors_= embed(model, anchors, reuse=reuse, training=True)
    positives_ = embed(model, positives, reuse=reuse, training=True)
    negatives_ = embed(model, negatives, reuse=reuse, training=True)

    loss = triplet_loss(anchors_, positives_, negatives_, alpha)
    trainer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    points = {
        'anchors': anchors,
        'positives': positives,
        'negatives': negatives,
        'alpha': alpha,
        'loss': loss,
        'train': trainer
    }

    return points


def face_train(model, run_name, max_iter=100, people=10, faces=30, batch_size=100,
               alpha=0.25, learning_rate=0.001, image_shape=None, dropout=0):

    if image_shape is None:
        images = provider.sample_people(num_people=1, num_faces=1)
        image_shape = (None, images[0].shape[1], images[0].shape[2], images[0].shape[3])

    global_step = tf.Variable(0, name='global_step', trainable=False)
    lr = tf.train.exponential_decay(learning_rate, global_step, 300, 0.96)

    reuse = os.path.exists(CHECKPOINT_LOC + run_name)

    inbound = tf.placeholder(tf.float32, image_shape)
    logits = embed(model, inbound, reuse, training=True)

    trainer = face_trainer(model, lr, image_shape, global_step)

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
            images = provider.sample_people(num_people=people, num_faces=faces, process=True)
            embeddings = []
            for i in range(len(images)):
                feed_dict = {inbound: images[i]}
                embeddings.append(sess.run(logits, feed_dict))

            embeddings = np.array(embeddings)
            triplet_idxs = provider.select_triplets(embeddings, hard_mine=False)
            del embeddings
            triplets = provider.build_batch(images, triplet_idxs)
            triplets = provider.shuffle_batch(triplets)

            if len(triplets) > batch_size > 0:
                triplets = triplets[:,:batch_size]

            train_dict = {
                trainer['anchors']: triplets[0],
                trainer['positives']: triplets[1],
                trainer['negatives']: triplets[2],
                trainer['alpha']: alpha
            }

            _, loss, step = sess.run([trainer['train'], trainer['loss'], global_step], feed_dict=train_dict)
            print iteration, step, loss

            if iteration % 10 == 0:
                saver.save(sess, CHECKPOINT_LOC + run_name + '/train.ckpt', global_step=global_step)

        saver.save(sess,  CHECKPOINT_LOC + run_name + '/train.ckpt', global_step=global_step)


def face_eval(model, run_name, images, image_shape):
    inbound = tf.placeholder(tf.float32, image_shape)
    logits = embed(model, inbound, reuse=True, training=False)
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

    return np.array(embeddings)
