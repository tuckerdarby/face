from core import face_train
from models import squeezenet
from models import inception_fn
import tensorflow as tf




# face_train(squeezenet.inference, '4_24', max_iter=500, learning_rate=0.001,
#            people=10, samples=10, image_shape=(None, 32, 32, 3))

face_train(inception_fn.inference, 'inception_tests', max_iter=3000, learning_rate=0.001,
           people=10, samples=14, image_shape=(None, 32, 32, 3))