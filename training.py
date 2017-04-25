from core import face_train
from models import squeezenet
from models import inception_fn


# face_train(squeezenet.inference, '4_23', max_iter=500, learning_rate=0.000001,
#            people=10, samples=30, image_shape=(None, 32, 32, 3))

face_train(inception_fn.inference, 'inception_tests', max_iter=5, learning_rate=0.001,
           people=3, samples=5, image_shape=(None, 32, 32, 3))