from core import face_train
from models import squeezenet
from models import inception_fn
import tensorflow as tf




# face_train(squeezenet.inference, 'test', max_iter=500, learning_rate=0.001,
#            people=2, samples=2, image_shape=(None, 32, 32, 3))
#
face_train(squeezenet.inference, 'squeeze', max_iter=10000, learning_rate=0.001, alpha=0.25, batch_size=100,
           people=40, samples=10, image_shape=(None, 32, 32, 3))

