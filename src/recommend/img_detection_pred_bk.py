import os
from recommend import EfficientNetModel
from utils.logging import Logger
logger = Logger(level='INFO')

MODEL_DIR = '/home/ubuntu/my_project/models/img_detection'
MODEL_PATH = os.path.join(MODEL_DIR, 'keras_efficientnet_w.h5')
LABEL_ENCODE_PATH = os.path.join(MODEL_DIR, 'label_encode.pkl')

class ImgDetectionPred:
    def __init__(self):
        """
        keras efficientnet-b0 ワイン画像分類用モデル
         - leakyReLU対応のため、load_modelではなくload_weightsでモデル生成        
        """
        self.eff = EfficientNetModel()
        self.model = self.eff.efficientnet_model(input_shape=(64,64,3), n_classes=2)
        self.model = self.eff.load_weights(self.model, MODEL_PATH)
    
    def predict(self, image):
        return self.eff.predict(self.model, image, LABEL_ENCODE_PATH)

