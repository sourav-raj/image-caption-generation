# -*- coding: utf-8 -*-
# Indentation: Jupyter Notebook

'''
read image captioning

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

def read_image_caption(filename:str):
    img_names, img_captionIds, img_captions=[], [], []
    for row in filename:
        cnt_=sum([1  if char =='#' else 0 for char in row ])
        if cnt_>1:
            print(row, cnt_)
        row=row.split('#')
        img_name, img_caption=row[0], row[1]
        captionId, caption = img_caption.split('\t')
        img_names.append(img_name.split('.jpg')[0])
        img_captionIds.append(captionId)
        img_captions.append(caption)

    image_captions=pd.DataFrame({'image':img_names, 'captionId':img_captionIds, 'caption':img_captions})
    return image_captions