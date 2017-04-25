import pandas as pd
import numpy as np
import os
import urllib
import socket
from PIL import Image
from constants import *


def detect_bad_image(image_location):
    try:
        Image.open(image_location).verify()
        return True
    except Exception:
        os.remove(image_location)
        print 'deleted - bad img detected'
        return False


def get_face(filename, url):
    if not os.path.exists(filename):
        try:
            f = urllib.urlretrieve(url, filename)
            print 'downloaded', filename, url
            valid_image = detect_bad_image(filename)
            return valid_image
        except Exception:
            print 'exception', filename, url
    else:
        print 'exists', filename
        # valid_image = detect_bad_image(filename)

    return False


def crop_image(img, left, top, right, bottom):
    cropped = img[top:bottom, left:right]
    return cropped


def crop_face(filename, left, top, right, bottom):
    try:
        img_arr = get_image_array(filename)
        cropped_arr = crop_image(img_arr, left, top, right, bottom)
        std = cropped_arr.std()
        if std < 1:
            os.remove(filename)
            print 'blank image/face - deleting file'
            return
        cropped_img = Image.fromarray(cropped_arr)
        os.remove(filename)
        cropped_img.save(filename)
        return cropped_arr, cropped_img
    except Exception:
        os.remove(filename)
        print 'bad crop - deleting file'


def get_image_array(image_location):
    img = Image.open(image_location)
    img.load()
    img_arr = np.asarray(img, dtype=np.uint8)
    return img_arr


socket.setdefaulttimeout(10)
filenames = os.listdir(LINK_LOC)
max_people = 100


def download_faces():
    for i, fn in enumerate(filenames):
        if i < 250:
            continue
        name = str(fn[:len(fn)-5]).strip().lower()
        df = pd.read_csv(LINK_LOC + fn, header=None, delimiter="\s+")

        print len(df)

        print name

        fn_dir = PEOPLE_LOC + name + '/'

        if not os.path.isdir(fn_dir):
            os.makedirs(fn_dir)

        for k in range(len(df)):
            url = str(df.ix[k][1])
            left, top, right, bottom = df.ix[k][2:6]
            ext = url[len(url)-4:].lower()
            if ext != '.jpg' and ext != '.jpeg' and ext != '.png':
                print 'skipping bad url', url
                continue

            filename = fn_dir + name + '_' + str(k) + ext

            valid_image = get_face(filename, url)

            if valid_image:
               crop_face(filename, left, top, right, bottom)

            if k >= max_people:
                break

        if i >= 300: break


download_faces()