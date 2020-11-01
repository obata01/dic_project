import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from utils.logging import Logger
logger = Logger(level='INFO')

K.clear_session()

IMG_SIZE = 112
# NUM_CLASSES = 3
NUM_CLASSES = 3954
# BATCH_SIZE = 4
BATCH_SIZE = 64
EPOCHS = 1
# IMG_PATH = '/home/ubuntu/my_project/src/data/wine'
IMG_PATH = '/home/ubuntu/my_project/data/raw/img/wine2'
# MODEL_PATH = '/home/ubuntu/my_project/models/img_detection/keras2_efficientnet_w.h5'
MODEL_PATH = '/home/ubuntu/my_project/models/img_detection/keras2_efficientnet_w2.h5'
train_dir = os.path.join(IMG_PATH, 'train')
validation_dir = os.path.join(IMG_PATH, 'validation')
# LABEL_PATH = '/home/ubuntu/my_project/models/img_detection/label_map.pkl'
LABEL_PATH = '/home/ubuntu/my_project/models/img_detection/label_map2.pkl'


class EfficientNetModel2:
    def __init__(self):
        self.model = None
        self.class_map = None

    
    def train(self):
        # Efficientnet
        logger.info('EfficientNet model loading start...')
        efficientnet_url = "https://tfhub.dev/google/efficientnet/b4/feature-vector/1"
        efficientnet_layer = hub.KerasLayer(efficientnet_url,
                                            input_shape=(IMG_SIZE,IMG_SIZE, 3))
        logger.info('EfficientNet model loading end...')

        # 学習済み重みは固定
        efficientnet_layer.trainable = False
        
        
        logger.info('Model built start...')
        model = tf.keras.models.Sequential()
        model.add(efficientnet_layer)
        # 3クラス用
#         model.add(layers.Dense(512, activation='relu'))
#         model.add(layers.Dropout(0.4))
#         model.add(layers.Dense(512, activation='relu'))
#         model.add(layers.Dropout(0.4))
#         model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

        # 全ワイン用
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss="categorical_crossentropy",
                      metrics=['mae', 'acc'])
        
        # Model load
        model = self.load_model()
        
        logger.info('Model built end')

        #loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True)

        model.summary()
        
        logger.info('Generate data set process start...')
        train_datagen = ImageDataGenerator(
                featurewise_center=False, 
                samplewise_center=False, 
                featurewise_std_normalization=False, 
                samplewise_std_normalization=False, 
                zca_whitening=False, 
                zca_epsilon=1e-06, 
                rotation_range=2.0, 
                width_shift_range=5.0, 
                height_shift_range=5.0, 
                brightness_range=None, 
                shear_range=0.0, 
                zoom_range=0.0, 
                channel_shift_range=60.0, 
                fill_mode='nearest', 
                cval=0.0, 
                horizontal_flip=False, 
                vertical_flip=False, 
                rescale=1.0/255, 
                preprocessing_function=None, 
                data_format=None, 
                validation_split=0.0
                )

        test_datagen = ImageDataGenerator(rescale=1.0 / 255)

        train_generator = train_datagen.flow_from_directory(
            train_dir,
            color_mode="rgb",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            validation_dir,
            color_mode="rgb",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='categorical')
        logger.info('Generate data set process end.')
        
        # 学習
        logger.info('Train start...')
        history = model.fit_generator(
                        train_generator, 
                        steps_per_epoch=int(NUM_CLASSES*2/BATCH_SIZE), 
                        validation_data=validation_generator, 
                        validation_steps=1,
                        epochs=EPOCHS,
                        shuffle=True)
        logger.info('Train end.')
        
        # インスタンス変数設定
        self.class_map = train_generator.class_indices
        self.model = model
        

        
        # モデル保存
        logger.info('Model save start...')
        model.save(MODEL_PATH)
        logger.info('Model save end.')
    
    
    def load_model(self):
        model = load_model(MODEL_PATH, custom_objects={"KerasLayer": hub.KerasLayer})
        return model
    

    
    
if __name__ == '__main__':
    ef = EfficientNetModel2()
    ef.train()
    
    
    try:
        logger.info('save label process start...')
        save_label(ef.class_map)
    except Exception as e:
        logger.error('save label error. {}'.format(e))
    else:
        logger.info('save label ended normally.')
        
        
    def save_label(data):
        with open('LABEL_PATH', 'wb') as f:
            pickle.dump(data, f)