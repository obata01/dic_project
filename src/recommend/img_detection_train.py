import os
from recommend import EfficientNetModel
from utils.logging import Logger
logger = Logger(level='INFO')


if __name__ == '__main__':
    # モデルクラスインスタンス化
    eff = EfficientNetModel()
    
    # config
    EPOCHS = 100
    BATCH_SIZE = 2
    IMG_SIZE = (64,64,3)
    IMG_PATH = '/home/ubuntu/my_project/data/raw/img/wine2/'
    CP_FILEPATH = '../models/weight/img_detection/20201027/cp-{epoch:04d}-01.ckpt'
    CP_DIR = os.path.dirname(CP_FILEPATH)
    LE_PATH = '/home/ubuntu/my_project/models/img_detection/label_encode.pkl'
    BK_SAVE_PATH = '../models/weight/img_detection/20201026/keras_efficientnet_w.h5'
    SAVE_PATH = '/home/ubuntu/my_project/models/img_detection/keras_efficientnet_w.h5'

    # data list
    train_df, n_samples, n_classes = eff.make_data_list(IMG_PATH)
    
    # image file check
#     img_channels_check(train_df, 'img'):
    
    # label encording
    labels = eff.labelEncodeAndOneHot(train_df, 'item_id', LE_PATH)
    
    # data set
    dataset = eff.data_set(train_df['img'], labels, eff.img_transformer, BATCH_SIZE)
    
    # model
    model = eff.efficientnet_model(input_shape=IMG_SIZE, n_classes=n_classes, base_trainable=False)
    
    # load weight
    model = eff.load_weights_cp(model, CP_DIR)
    
    # fit(model, dataset, checkpoint_path, epochs)
    eff.fit(model, dataset, CP_FILEPATH, EPOCHS)
    
    # save model
#     eff.save_weights(model, BK_SAVE_PATH)
    eff.save_weights(model, SAVE_PATH)

