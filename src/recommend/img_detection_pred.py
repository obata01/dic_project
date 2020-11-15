#!/usr/bin/env python3

# ワイン画像推定するための処理

import os
import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from PIL import Image
import pickle
import matplotlib.pyplot as plt
from utils.logging import Logger
logger = Logger(level='DEBUG')


IMG_SIZE = 112
NUM_CLASSES = 7
# NUM_CLASSES = 3954
MODEL_PATH = '/home/ubuntu/my_project/models/img_detection/keras2_efficientnet_w3.h5'
LABEL_PATH = '/home/ubuntu/my_project/models/img_detection/label_map3.pkl'
# MODEL_PATH = '/home/ubuntu/my_project/models/img_detection/keras2_efficientnet_w2.h5'
# LABEL_PATH = '/home/ubuntu/my_project/models/img_detection/label_map2.pkl'

class ImgDetectionPred:
    def __init__(self):
        self.model = self.load_model()
        self.class_map = {v: k for k, v in self.load_labels().items()}
        
        
    def predict(self, img_path):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            img = Image.open(img_path)
            img = np.asarray(img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)))[np.newaxis, :, :, :] / 255
            pred = self.model.predict_proba(img)[0]
            idx = np.argmax(pred)
            logger.info('Predict results : {}'.format(pred))
            class_id = str(self.class_map[idx])
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
            sys.exit(1)
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))
            logger.info('class_id = {}, pred = {}'.format(class_id, pred[idx]))
            return class_id, pred[idx]
    

    def load_model(self):
        model = load_model(MODEL_PATH, custom_objects={"KerasLayer": hub.KerasLayer})
        return model
    
    
    def load_labels(self):
        with open(LABEL_PATH, 'rb') as f:
            data = pickle.load(f)
        return data
