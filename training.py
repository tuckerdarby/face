from core import face_train
from models import squeezenet
from models import inception_fn
import tensorflow as tf


face_train(squeezenet.inference, 'squeeze_test2', max_iter=5, learning_rate=0.001,
           people=25, faces=25, image_shape=(None, 32, 32, 3))
#
# face_train(inception_fn.inference, 'inception_test', max_iter=10000, learning_rate=0.001, alpha=0.25, batch_size=100,
#            people=20, faces=20, image_shape=(None, 32, 32, 3), dropout=0.05)

