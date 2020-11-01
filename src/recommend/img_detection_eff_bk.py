import os
import gc
import re
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.utils import to_categorical
# from keras_radam import RAdam
from tensorflow.keras import backend as K
from utils.logging import Logger
logger = Logger(level='INFO')

K.clear_session()

class EfficientNetModel:
    def __init__(self):
        pass

    def efficientnet_model(self, input_shape, n_classes, base_trainable=False):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        initializer = tf.keras.initializers.HeNormal()
        base_model = EfficientNetB0(input_shape=input_shape, weights='imagenet', include_top=False)
        base_model.trainable = base_trainable
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(2024, activation=layers.LeakyReLU(), kernel_initializer=initializer, bias_initializer='zeros')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(2048, activation=layers.LeakyReLU(), kernel_initializer=initializer, bias_initializer='zeros')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(4096, activation=layers.LeakyReLU(), kernel_initializer=initializer, bias_initializer='zeros')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(n_classes, activation='sigmoid')(x)
        model = keras.Model(inputs=base_model.input, outputs=x)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        return model



    def img_transformer(self, filename, label, mode='train'):
        image = tf.image.decode_jpeg(tf.io.read_file(filename))
        if mode == 'train':
            image = tf.image.adjust_brightness(image, 0.1)
            image = tf.image.resize_with_pad(image, target_height=112, target_width=112)
            image = tf.image.random_crop(image, (112,112,3))
            label = tf.cast(label, tf.float32)
        elif mode == 'test':
            image = tf.image.resize_with_pad(image, target_height=112, target_width=112)
            image = tf.image.random_crop(image, (112,112,3))
        image /= 255
        image = tf.cast(image, tf.float32)
        return image, label


    def data_set(self, img_lists, labels, transform, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((img_lists, labels))
        dataset = dataset.map(transform)
        dataset = dataset.batch(batch_size)
        dataset = dataset.shuffle(1000)
        return dataset


    def labelEncodeAndOneHot(self, df, column_name, save_path):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            # label encode
            le = LabelEncoder()
            df_ = df.copy()
            df_['labels'] = le.fit_transform(df_[column_name])

            # one-hot
            labels = to_categorical(df_['labels'])

            # save label encode
            with open(save_path, 'wb') as f:
                pickle.dump(le, f)

            del df_
            gc.collect()
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))
            return labels

        
    def fit(self, model, dataset, checkpoint_path, epochs):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            # check point callback
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path,
                verbose=1,
                save_weights_only=True,
                period=100)

            # save weight
            model.save_weights(checkpoint_path.format(epoch=0))

            # train
            model.fit(dataset, 
                      epochs=epochs,
                      callbacks=[cp_callback],
                      shuffle=True)    
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
            sys.exit(1)
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))
            
            

    def predict(self, model, img_data, label_path):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            # label load
            with open(label_path, 'rb') as f:
                le = pickle.load(f)
            # resize
            img, _ = self.img_transformer(img_data, label='', mode='test')
            # predict
            pred = model.predict(img[np.newaxis, :, :, :])
            class_idx = np.argmax(pred)
            class_id = le.inverse_transform([class_idx])
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))
            return str(class_id[0]), pred[0][class_idx]
        
        


    def make_data_list(self, img_path):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            # data list
            train_df = pd.DataFrame()
    #         train_df['img'] = glob.glob(img_path + '*.[jp][pn]g')[:30]
            train_df['img'] = glob.glob(img_path + '*.[jp][pn]g')
            train_df['item_id'] = train_df['img'].apply(lambda x: re.sub('(.jpg|_2.png)', '', x[42:]))
#             train_df = train_df[:1800]
            n_samples = len(train_df)

            # number of classes
            n_classes = train_df['item_id'].unique().shape[0]
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
        else:
            logger.info('n_samples : {}'.format(n_samples))
            logger.info('n_classes : {}'.format(n_classes))
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))
            return train_df, n_samples, n_classes



    def img_channels_check(self, df, col_name):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            for i in df[col_name]:
                with Image.open(i) as img:
                    ndarray_img = np.array(img)
                    if ndarray_img.shape[-1] < 3:
                        logger.warning(nd.shape, i)
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))   

            
    def load_weights_cp(self, model, checkpoint_dir):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            model.load_weights(latest)
            logger.info('CheckPoint File is {}'.format(latest))
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
            sys.exit(1)
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))
            return model
        
        
    def load_weights(self, model, weights_path):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
#             model = tf.keras.models.load_weights(weights_path)
            model.load_weights(weights_path)
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
            sys.exit(1)
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))
            return model 


    def load_model(self, model_path):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
            sys.exit(1)
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))
            return model


    def save_model(self, model, model_dir):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            model.save(model_dir)
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))

            
            
    def save_weights(self, model, weights_dir):
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            model.save_weights(weights_dir)
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))
