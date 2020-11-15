#!/usr/bin/env python3

# Doc2Vecを使用したワイン説明文のレコメンドで
# 必要な処理をまとめたクラス


import os
from models.common import Doc2Vec
from models.common import JanomeTokenizer
from utils import Logger

class ContentsRecommend():
    def __init__(self, model_load):
        # Tokenizer 初期化
        janome = JanomeTokenizer(part_of_speech=['名詞', '形容詞'])

        # Doc2Vec 初期化
        input_path = '/home/ubuntu/my_project/src/data/wine_desc3_ajiwai'
        model_dir = '/home/ubuntu/my_project/src/models/weight/contents_recommend/doc2vec'
        model_file_name = '20201025_d2v.model'
        self.model = Doc2Vec(tokenizer=janome, 
                             input_path=input_path, 
                             model_path=os.path.join(model_dir, model_file_name), 
                             model_load=model_load
                             )
        
    def fit(self, epochs):
        """トレーニング"""
        self.model.fit(epochs)
    
    
    def id_predict(self, doc_id, topn):
        """item_idからレコメンド商品を抽出"""
        return self.model.sentences_similar_to_id(doc_id, topn)
    
    
    def doc_predict(self, doc, topn):
        """任意の文章から類似する商品を抽出"""
        return self.model.sentences_similar_to_free_words(doc, topn)
    
    
    def word_predict(self, word, topn):
        """テスト用 単語から類似単語を抽出"""
        return self.model.word_similar_to_word(word, topn)
    
    def to_vec(self, word):
        """テスト用 単語をベクトル化"""
        return self.model.get_vec(word)
    
    def vec2word(self, vec, topn):
        """テスト用 ベクトル値を単語化"""
        return self.model.vec_to_similar_word(vec, topn)
    

        
