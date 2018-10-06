# !/usr/bin/python 
# -*-coding:utf-8 -*- 
import sys, os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# from scipy.misc import imread, imsave, imresize
from scipy.misc import imread, imsave, imresize
import time, datetime, random
import re, requests, json
import numpy as np
from keras.models import Model, load_model

def getKeybyVal(dictionary,idx):
    return list(dictionary.keys())[list(dictionary.values()).index(idx)]


if __name__ == "__main__":
    BASE_DIR = "data/valid"
    dictionary = json.load(open('./plants_dict.json'))
    MODEL_PATH = sys.argv[1]
    model = load_model(MODEL_PATH)
    hit = 0
    miss = 0
    # traverse valid data directory
    for root, dirs, files in os.walk(BASE_DIR):
        path = root.split(os.sep)
        plant = os.path.basename(root)
        if plant not in dictionary:
            continue
        label = dictionary[plant]
        print(plant)
        for file in files:
            img = imread( os.path.join(BASE_DIR, plant, file), mode ='RGB')
            img = imresize(img ,size=(224,224))
            img = np.asarray(img)
            prob = model.predict(np.expand_dims(img, axis=0))
            # get top 5
            sorted_idx = list(np.argsort(prob[0]))
            sorted_idx = sorted_idx[::-1]
            top_result = getKeybyVal(dictionary, sorted_idx[0])
            top5_idx = sorted_idx[0:5]
            if label in top5_idx:
                hit = hit + 1
            else:
                miss = miss + 1
    print('Hit: %s, Miss: %s, Accuracy: %s' % (hit, miss, hit/(hit+miss)))