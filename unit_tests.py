import tensorflow as tf
import aid
from PIL import Image
import urllib, urllib2
import requests
import socket

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


def skip_authentication():
    good = "http://static2.vidics.ch/150/11/Spencer_Boldman-216642.JPEG"
    bad = "http://www.kinkyceleb.celebarazzi.com/content/Portraits/N/Nicholle_Tom/portrait.jpg"
    socket.setdefaulttimeout(10)

    def download2(url, name):
        print 'attempting to download', name, url
        try:
            response = urllib2.urlopen(url, timeout=2)
            with open(name, 'wb') as f:
                f.write(response.read())
                print 'saved', name
            print 'downloaded', name
        except urllib2.HTTPError, x:
            print 'ignoring', name


    def download(url, name):
        print 'attempting to download', name, url
        try:
            f = urllib.urlretrieve(url, name)
            print 'downloaded', name
        except urllib.error.HTTPError, x:
            print 'ignoring', name


    def request(url, name):
        print 'requesting', name
        r = requests.get(url, verify=False,  timeout=3)
        with open(name, "wb") as f:
            print 'writing', name
            f.write(r.content)

    download2(good, 'good')
    download2(bad, 'bad')

skip_authentication()
# test_squeezenet()

