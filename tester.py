import numpy as np
import seaborn as sns
import provider
from models import squeezenet
from models import inception_fn
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


def print_differences(embeddings):
    size = len(embeddings)
    for i in range(size):
        print i, '-----------'
        correct = False
        for k in range(size):
            t, d, m = get_differences(embeddings[i][:5], embeddings[k][5:])
            print i, k, m, t
            t, d, m = get_differences(embeddings[i][5:], embeddings[k][:5])
            print i, k, m, t
        print '-------------'


def difference_map(embeddings):
    size = len(embeddings)
    diff = np.array([[0 for _ in range(size)] for _ in range(size)])
    for i in range(size):
        for k in range(size):
            t, ds, m = get_differences(embeddings[i][:5], embeddings[k][5:])
            diff[i, k] = m * 100
    return diff


def ranked_differences(embeddings):
    size = len(embeddings)
    correct = 0
    guesses = []
    for i in range(size):
        ms = []
        for k in range(size):
            t, ds, m = get_differences(embeddings[i], embeddings[k])
            ms.append(m)
        closest = np.argmin(ms)
        guesses.append(closest)
        if closest == i:
            correct += 1
    acc = np.divide(correct * 1.0, size)
    return guesses, correct, acc


def test_model(model, name):

    images = provider.sample_people(samples=10, num_people=10, process=True)
    image_shape = None, images[0].shape[1], images[0].shape[2], images[0].shape[3]
    embeddings = face_eval(model, name, images, image_shape)

    diffs = difference_map(embeddings)
    guesses, correct, acc = ranked_differences(embeddings)

    print diffs
    print correct, acc
    # print_differences(embeddings)


test_model(squeezenet.inference, 'squeeze')
