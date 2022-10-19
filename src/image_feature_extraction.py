# -*- coding: utf-8 -*-
# Indentation: Jupyter Notebook

'''
to extract image feature

'''

__version__ = 1.0
__author__ = "Sourav Raj"
__author_email__ = "sourav.raj@ril.com"

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import random
import warnings
warnings.filterwarnings('ignore')

import matplotlib.image as img

#Import Necessary Libraries
from os import listdir
from pickle import dump, load
import pickle 

from tqdm.notebook import tqdm

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model

def image_feature_extraction(image_directory, model):
    '''
        extract image features from the given directory using pretrained model
    '''
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    print(model.summary())
    # extract features from each photo
    features = dict()
    for name in tqdm(listdir(image_directory)):
        # load an image from file
        filename = image_directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        # reshape data for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # prepare the image for the given model
        image = preprocess_input(image)
        # get features
        feature = model.predict(image, verbose=0)
        # get image id
        image_id = name.split('.')[0]
        # store feature
        features[image_id] = feature
#         print('>%s' % name)
    return features