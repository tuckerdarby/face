import tensorflow as tf
import numpy as np
import aid
import os
import processor
from constants import *


def sample_people(samples=0, num_people=0, process=True):
    people_names = os.listdir(RECORD_LOC)
    if num_people <= 0:
        num_people = len(people_names)

    people_data = []
    idxs = np.random.permutation(range(len(people_names)))[:num_people]
    # sample_names = np.random.choice(people_names, num_people)
    sample_names = np.array(people_names)[idxs]

    for i, name in enumerate(sample_names):
        data = aid.extract_data(name)
        if process:
            data = processor.whiten(data)
        num_data = len(data)
        if samples == 0:
            num_samples = num_data
        else:
            num_samples = min(samples, num_data)
        sample_indicies = np.random.choice(range(num_data), num_samples)

        sample_data = data[sample_indicies]
        people_data.append(sample_data)

    return people_data


def negative_distances(embeddings, pos_class_idx, anchor_idx):
    neg_distances = []
    anchor = embeddings[pos_class_idx][anchor_idx]
    for class_idx in range(pos_class_idx+1, len(embeddings)):
        negs = np.sum(np.square(anchor - embeddings[class_idx]), axis=1)
        neg_distances.extend(negs)
    return np.array(neg_distances)


def select_triplets(embeddings, hard_mine=False):
    num_people = embeddings.shape[0]
    num_faces = embeddings.shape[1]
    num_choices = num_faces * num_people

    triplets = []
    vals = []
    for i in range(num_people):
        for a in range(num_faces-1):
            anchor = embeddings[i, a]

            neg_dists = np.square(anchor-embeddings)
            neg_dists = np.sum(np.square(anchor - embeddings), axis=2)
            neg_dists[i] = np.nan

            for p in range(a+1, len(embeddings[i])):
                positive = embeddings[i, p]
                post_dist = np.sum(np.square(anchor-positive))

                dists = neg_dists - post_dist
                dists = dists.flatten()

                if hard_mine:
                    furthest = np.argsort(dists)[0]
                else:
                    choice = np.random.randint(0, min(max(15, int(num_choices*0.1)), num_choices))
                    furthest = np.argsort(dists)[choice]
                dist = dists[furthest]
                vals.append(dist)
                neg_idx = furthest / num_faces
                neg_pos = furthest % num_faces
                triplet = [(i, a), (i, p), (neg_idx, neg_pos)]
                triplets.append(triplet)

    ranked = np.argsort(vals)
    triplets = np.array(triplets)[ranked]
    return triplets


def build_batch(images, triplets):
    batch = [[], [], []]
    for triplet in triplets:
        for i, coord in enumerate(triplet):
            batch[i].append(images[coord[0]][coord[1]])

    for k in range(3):
        batch[k] = np.array(batch[k])

    batch = np.array(batch)
    return batch


def shuffle_batch(batch):
    args = np.random.permutation(range(len(batch[0])))
    shuffled = batch[:, args]
    return shuffled


def test_triplets(classes=3, size=5):
    embeddings = []
    for c in range(classes):
        embs = np.random.random_integers(0, 10, size * classes).reshape(classes, size)
        embeddings.append(embs)
    embeddings = np.array(embeddings)

    triplets = select_triplets(embeddings)
    print '---triplets---'
    print triplets
