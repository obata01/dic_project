import glob
import re
import os
import sys
import numpy as np
from tqdm import tqdm
import collections
from gensim import models
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from utils import Logger
logger = Logger(level='INFO')


class Doc2Vec:
    def __init__(self, tokenizer, input_path, model_path, model_load=False):
        self.tokenizer = tokenizer
        self.input_path = input_path
        self.model_path = model_path
        
        # 学習履歴表示用
        self.data_dict = {}
        self.history = []
        self.test_vec = {}
        
        # 既存のベクトル値を使用する場合のモデル初期化処理
        self.__model = None
        if model_load:
            logger.info('Model load start...')
            try:
                self.__model = models.Doc2Vec.load(self.model_path)
            except Exception as e:
                logger.error('Model load error. {}'.format(e))
                sys.exit(1)
            else:
                logger.info('Moel load ended normally.')
        

    def __preprocessing(self, text):
        """テキスト前処理"""
        # urlを除去
        text = re.sub(r'https?://[\w/:%#\$&\?\(\)~\.=\+\-]+', '', text)
        
        # 改行、スペースを削除
        text = re.sub('\n', '', text)
        text = re.sub(' ', '', text)
        text = re.sub('　', '', text)
        
        # 記号削除
        del_symbols = re.compile(
            '[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＜＞＠。、？！｀＋￥％～]')
        text = del_symbols.sub('', text)
        
        # 数字文字削除
        text = re.sub(r'[0-9 ０-９]', '', text)
        
        return text
    
    
    def __split_into_words(self, doc, name=''):
        """形態素解析"""
        words = self.__del_vocab(self.tokenizer(doc))
        return TaggedDocument(words=words, tags=[name])
    
    
    def __del_vocab(self, words):
        """特定の文字列を削除"""
        del_list = ['1本', 'ワイン', '味', '味わい', '果実', '年', 'こちら', 'さ', ]
        words = [w for w in words if w not in del_list ]
        return words
    
    
    def __read_files(self, character_limit, encode='utf8'):
        """ディレクトリ内のtxtファイルを取得して辞書{item_id:doc}で保持"""
        all_files = glob.glob(self.input_path + '/*.txt')
        for file_ in all_files:
            item_id = os.path.splitext(os.path.basename(file_))[0]
            with open(file_, 'r', encoding=encode) as f:
                data = self.__preprocessing(f.read())[:character_limit]
                if len(data) < 150:
                    pass
                else:
                    self.data_dict[item_id] = data
        return self.data_dict
    
    
    
    def __corpus_to_sentences(self, data_dict):
        """コーパスをドキュメント別かつ単語別にして返却するジェネレータ"""
        for idx, (name, doc) in enumerate(data_dict.items()):
            yield self.__split_into_words(doc, name)
    
    
    
    def __train(self, epochs):
        """学習用"""
        logger.info('Reading files and text preprocessing start...')
        try:
            corpus = self.__read_files(character_limit=1000)
            sentences = list(self.__corpus_to_sentences(corpus))
        except Exception as e:
            logger.error('Reading files and text preprocessing error. {}'.format(e))
            sys.exit(1)
        else:
            logger.info('Reading files and text preprocessing ended normally.')
            

        # 学習
        logger.info('Doc2Vec model training start...')
        try:
            #
            # -- models.Doc2Vec --
            # dm : 1=PV-DM, 0=PV-DBOW で学習(語順に意味があれば1,無ければ0)
            # venctor_size : 分散表現の次元数
            # window : ウインドウサイズ
            # min_count : 出現回数の足切りライン
            # wokers : 学習に用いるスレッド数
            # 詳細 -> https://radimrehurek.com/gensim/models/doc2vec.html
            #
            self.__model = models.Doc2Vec(sentences, dm=1, window=5, alpha=0.03, epochs=1000,
                         min_alpha=0.05, min_count=200, sample=1e-5, negative=5, wokers=1)
            for epoch in range(epochs):
                logger.info('----------------------------------------------------------------')
                logger.info('Epoch: {}/{} start...'.format(epoch+1, epochs))
                self.__model.train(sentences, 
                                 total_examples=sum([len(s) for s in sentences]), 
                                 epochs=self.__model.iter)
                # テストスコア表示
                self.__test_proc()
        except Exception as e:
            logger.error('Doc2Vec model training error. {}'.format(e))
            sys.exit(1)
        else:
            logger.info('Doc2Vec model training ended normally.')
        
        # 学習結果保存
        logger.info('Model save process start...')
        try:
            self.__model.save(self.model_path)
        except Exception as e:
            logger.error('Model save process error. {}'.format(e))
        else:
            logger.info('Model save process ended normally.')
    
    
    
    def __get_similar_docs(self, doc, topn):
        """類似するドキュメントを返却"""
        logger.info('Search similar document process start...')
        try:
            results = []
            similar_docs = self.__model.docvecs.most_similar(doc, topn=topn)
        except Exception as e:
            logger.error('Search similar document process error. {}'.format(e))
            return
        else:
            logger.info('Search similar document process ended normally.')
            return similar_docs
#             return results


    def __get_similar_word(self, word, topn):
        """類似する単語を返却"""
        logger.info('Search similar word process start...')
        try:
            similar_word = self.__model.similar_by_word(word, topn=topn)
        except Exception as e:
            logger.error('Search similar word process error. {}'.format(e))
            return
        else:
            logger.info('Search similar word process ended normally.')
            return similar_word

        
    def __cos_sim(self, v1, v2):
        """コサイン類似度を算出"""
        logger.info(v1)
        logger.info(v2)
        v1 *= 100
        v2 *= 100
        logger.info(v1)
        logger.info(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    
    def __test_proc(self, test_size=20):
        """学習履歴（同一文書同士のコサイン類似度を計算）"""
        if self.test_vec != {}:
            self.__test_score(self.test_vec)
            
        test_data = np.random.choice(list(self.data_dict.keys()), test_size, replace=False)
        test_data = ['021920270', 'FA1240060', '320054312']
        self.test_vec = self.__current_vector(test_data)
        
        
    def __current_vector(self, test_data):
        """入力の文書ID毎のベクトル値をディクショナリで返却"""
        test_dict = {}
        for doc_id in test_data:
            test_dict[doc_id] = self.__model.infer_vector([doc_id])
        return test_dict
    
          
    def __test_score(self, previous_dict):
        """前回エポックで算出したベクトルを使用してコサイン類似度を計算"""
        cos_sim = 0
        for doc_id, vec1 in previous_dict.items():
            vec2 = self.__model.infer_vector([doc_id])
            value = self.__cos_sim(vec1, vec2)
            cos_sim += value
            logger.info('doc_id : {}, cos_sim = {}'.format(doc_id, value))
        cos_sim /= len(previous_dict)
        self.history.append(cos_sim)
        logger.info('Average cos_sim = {}'.format(cos_sim))

    
    
    def fit(self, epochs):
        """学習実行"""
        self.__train(epochs)
    
    
    def sentences_similar_to_free_words(self, doc, topn):
        """任意の文章から類似したドキュメントを返却"""
        words = self.__split_into_words(doc)
        words_vec = self.__model.infer_vector(words[0])
        return self.__get_similar_docs([words_vec], topn=topn)
    
    
    def sentences_similar_to_id(self, doc_id, topn):
        """指定したIDに類似するドキュメントを返却"""
        return self.__get_similar_docs(doc_id, topn=topn)
    
    
    def word_similar_to_word(self, word, topn):
        """特定の単語に類似する単語を返却"""
#         word = self.__split_into_words(word)[0][0]
        return self.__get_similar_word(word, topn=topn)
    
    def get_vec(self, word):
        """特定の単語・文章の分散値を返却"""
        word = self.__split_into_words(word)[0]
        return self.__model.infer_vector(word)
    
    def vec_to_similar_word(self, vec, topn):
        """分散値から類似の単語を返却"""
        return self.__get_similar_word(vec, topn=topn)
    

    def get_dataframe(self):
        """学習した単語や頻度をデータフレームで返却"""
        data_list = []
        for w, v in self.__model.wv.vocab.items():
            data_list.append([w, v.count])
        df = pd.DataFrame(data_list, columns=['word', 'count'])
        return df.sort_values(by='count', ascending=False)
    
    
    def tsne(self):
        vocabs = self.__model.wv.vocab.keys()
        print(vocabs)
        tsne_model = TSNE(perplexity=50, n_components=2, init="pca", n_iter=3000, random_state=23)
        vectors_tsne = tsne_model.fit_transform(self.__model[vocabs])
        fig, ax = plt.subplots(figsize=(15,15))
        ax.scatter(vectors_tsne[:, 0], vectors_tsne[:, 1])
        for i, word in enumerate(list(vocabs)):
            plt.annotate(word, xy=(vectors_tsne[i, 0], vectors_tsne[i, 1]))
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        plt.show()

        

        