#!/usr/bin/env python3

# ワイン説明文レコメンドの推論用処理

from recommend import ContentsRecommend


class ContentsRecommendPred:
    """自然言語処理レコメンド返却"""
    def __init__(self):
        self.model = ContentsRecommend(model_load=True)
    
    def id_predict(self, item_id, topn=10):
        """item_id から類似商品を抽出"""
        return self.model.id_predict(item_id, topn=topn)
    
    def doc_predict(self, doc, topn=10):
        """任意の文章から類似商品を抽出"""
        return self.model.doc_predict(doc, topn=topn)
    
    def word_predict(self, word, topn=1):
        """テスト用 単語から類似単語を抽出"""
        return self.model.word_predict(word, topn=topn)
    
    def to_vec(self, word):
        """テスト用 単語をベクトル化"""
        return self.model.to_vec(word)
    
    def vec2word(self, vec, topn):
        """テスト用 ベクトル値を単語化"""
        return self.model.vec2word(vec, topn)
    