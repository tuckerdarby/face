import numpy as np
import provider
from models.squeezenet import inference
from core import face_eval


def get_differences(a, b):
    total = 0
    ds = []
    for i in range(len(a)):
        for k in range(len(b)):
            d = np.sum(np.square(a[i] - b[k]))
            total += d
            ds.append(d)
    return total, ds, np.array(ds).mean()


def test_model(name='4_23'):

    images = provider.sample_people(samples=10, num_people=10, process=True)
    image_shape = None, images[0].shape[1], images[0].shape[2], images[0].shape[3]
    embeddings = face_eval(inference, name, images, image_shape)

    print len(embeddings)


test_model()
