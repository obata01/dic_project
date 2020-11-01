import os
import gc
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
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K

K.clear_session()

        
class EfficientNetModels(tf.keras.Models):
    def __init__(self, input_shape, n_classes, base_trainable=False):
        initializer = tf.keras.initializers.HeNormal()
        base_model = EfficientNetB4(input_shape=input_shape, weights='imagenet', include_top=False)
        base_model.trainable = base_trainable
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1024, activation=layers.LeakyReLU(), kernel_initializer=initializer, bias_initializer='zeros')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(1024, activation=layers.LeakyReLU(), kernel_initializer=initializer, bias_initializer='zeros')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(1024, activation=layers.LeakyReLU(), kernel_initializer=initializer, bias_initializer='zeros')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(1024, activation=layers.LeakyReLU(), kernel_initializer=initializer, bias_initializer='zeros')(x)
        x = layers.Dropout(rate=0.3)(x)
        x = layers.Dense(n_classes, activation='sigmoid')(x)
        model = keras.Model(inputs=base_model.input, outputs=x)
        return model



    def img_transformer(self, filename, label):
        image = tf.image.decode_jpeg(tf.io.read_file(filename)) # ファイル名 => 画像
        image = tf.image.adjust_brightness(image, 0.1)
        image = tf.image.resize_with_pad(image, target_height=self.img_size[0], target_width=self.img_size[1])
        image = tf.image.random_crop(image, self.img_size)
        image /= 255
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        return image, label



    def data_set(self, img_lists, labels, transform):
        dataset = tf.data.Dataset.from_tensor_slices((img_lists, labels))
        dataset = dataset.map(transform)
        dataset = dataset.batch(self.batch_size)
    #     dataset = dataset.repeat(1)
        dataset = dataset.shuffle(1000)
        return dataset
    
    
    def labelEncodeAndOneHot(self, df, column_name, save_path):
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
        return labels

        
with open('foo.p', 'wb') as f:
  pickle.dump(le, f)
with open('foo.p', 'rb') as f:
  le2 = pickle.load(f)
        pass
    
    
    def predict(self):
        # load label encode
        
        pass



def main():
    # config
    EPOCHS = 10
    BATCH_SIZE = 2
    IMG_SIZE = (224,224,3)
    IMG_PATH = '/home/ubuntu/my_project/data/raw/img/wine/' 
    CP_FILEPATH = '/home/ubuntu/my_project/src/models/weight/img_detection/20201023/cp-{epoch:04d}.ckpt'
    CP_DIR = os.path.dirname(CP_FILEPATH) 

    # data list
    train_df = pd.DataFrame()
    train_df['img'] = glob.glob(IMG_PATH + '*.[jp][pn]g')[:30]
    train_df['item_id'] = train_df['img'].apply(lambda x: x[42:51])
    n_samples = len(train_df)

    # label encording + one-hot
    le = LabelEncoder()
    train_df['labels'] = le.fit_transform(train_df['item_id'])
    labels = to_categorical(train_df['labels'])

    # number of classes
    n_classes = train_df['item_id'].unique().shape[0]

    # data set
    dataset = data_set(train_df['img'], labels, img_transformer, BATCH_SIZE)

    # model
    imd = ImageDetection(BATCH_SIZE, EPOCHS, IMG_SIZE)
    model = efficientnet_model(input_shape=IMG_SIZE, n_classes=n_classes, base_trainable=False)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    # load weight
    latest = tf.train.latest_checkpoint(CP_DIR)
    model.load_weights(latest)

    # check point callback
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CP_FILEPATH,
        verbose=1,
        save_weights_only=True,
        period=5)

    # save
    model.save_weights(CP_FILEPATH.format(epoch=0))

    # train
    model.fit(dataset, 
              epochs=EPOCHS,
              callbacks=[cp_callback],
              shuffle=True)

if __name__ == '__main__':
    main()
