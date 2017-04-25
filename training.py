from core import face_train
from models import squeezenet


face_train(squeezenet.inference, '4_23', max_iter=500, learning_rate=0.000001,
           people=10, samples=30, image_shape=(None, 32, 32, 3))
