import tensorflow as tf
import aid
from PIL import Image


def test_squeezenet():
    from models import squeezenet

    images = aid.read_data('data/test')
    image_shape = (None, images.shape[1], images.shape[2], images.shape[3])


    inbound = tf.placeholder(tf.float32, image_shape)
    logits_ = squeezenet.inference(inbound, 0.9)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        feed_dict = {inbound: images}
        logits = sess.run([logits_], feed_dict)

    return logits

# test_squeezenet()

