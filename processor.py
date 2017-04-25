import tensorflow as tf
import numpy as np


def whiten(batch):
    processed = []
    for sample in batch:
        mean = sample.mean()
        std = sample.std()
        std_adj = np.maximum(std, 1.0/np.sqrt(sample.size))
        whitened = np.multiply(np.subtract(sample, mean), 1/std_adj)
        processed.append(whitened)
    processed = np.array(processed)
    return processed

