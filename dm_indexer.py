from __future__ import division

import pandas as pd
import numpy as np
from PIL import Image
from core import face_eval
from constants import *


def index_dm(model, name, batch=50):
    embs = []
    df = pd.read_csv(DM_DATA_LOC + 'index.csv')
    df = df[df['cropped'] == 1]
    df.to_csv(DM_DATA_LOC + 'cropped_index.csv')
    amount = len(df)
    batches = amount / batch

    for i in range(batches):
        start = i*batch
        end = (i+1)*batch
        rows = df.ix[start:end]
        images = get_images(rows)
        image_shape = None, 32, 32, 3
        embeddings = face_eval(model, name, images, image_shape)
        embs.extend(embeddings)

    emb_df = pd.DataFrame(embs)
    emb_df.to_csv(DM_DATA_LOC + 'cropped_embeddings.csv')


def get_images(rows, ext='.jpg'):
    images = []
    for row in rows.iterrows():
        identity = row['identity']
        filename = DM_PEOPLE_LOC + identity + '/faces/' + identity + ext
        img = Image.open(filename)
        img.load()
        img_arr = np.asarray(img, dtype=np.uint8)
        images.append(img_arr)
    return images
