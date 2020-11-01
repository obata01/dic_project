import ast
from django.db import models
from recommend import ImgDetectionPred
import io, base64
import tensorflow as tf
from PIL import Image
import mysql.connector
from db import DBHandler
from db import RecommendDB
from recommend import ContentsRecommendPred
from utils.logging import Logger
logger = Logger(level='DEBUG')


# モデル初期化
i_model = ImgDetectionPred()
kw_model = ContentsRecommendPred()

# 商品詳細検索用カラムインデックス番号
ITEM_TABLE_IDX1 = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12, 13]
# レコメンド商品検索用カラムインデックス番号
ITEM_TABLE_IDX2 = [0, 1, 2, 3, 4, 6, 10, 12]
# SELECT文生成用関数
def generate_sql(table, idxs, n_where):
    rdb = RecommendDB()
    item_table = rdb.columns[table]
    s = "SELECT "
    for idx in idxs:
        if idx != 0:
            s += ', '
        s += item_table[idx]
    s += ' FROM ' + table + ' WHERE id IN ('
    for i in range(n_where):
        if i != 0:
            s += ', '
        s += '%s'
    s += ')'
    return s



class Recommend(models.Model):
    id = models.CharField(max_length=16, primary_key=True, null=False)
    value = models.CharField(max_length=512, null=False)
    
    class Meta:
        managed = False
        db_table = 'recommend'

        
class Photo(models.Model):
    image = models.ImageField(upload_to='images')
    fpath = './media/wine_test.jpg'
    sql = generate_sql(table='item', idxs=ITEM_TABLE_IDX1, n_where=1)
    
    
    def predict(self):
        logger.debug('models.py - photo class -  predict start...')
        
        # 画像データ処理
        img_data = self.image.read()
        img_bin = io.BytesIO(img_data)
        
        # jpg形式で保存
        with Image.open(img_bin) as img:
            img.save(self.fpath)
            
        # predict実行
        item_id, proba = i_model.predict(self.fpath)
        logger.info('item_id : '.format(item_id))
        logger.info('proba : '.format(proba))
        
        # SQL実行
        sql_result = self.db_select(self.sql, (item_id,))
        
        # レコメンド取得
        recommend_items = self.recommend(item_id)
        
        return item_id, proba, sql_result[0], recommend_items
    
    
    def image_src(self):
        """画像表示用にデコード・エンコードする処理"""
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()
            return 'data:' + img.file.content_type + ';base64,' + base64_img
        
        
    def db_select(self, sql, values):
        """SELECT文実行"""
        # SQL実行処理
        logger.info('DB connect process start...')
        logger.debug(sql)
        logger.debug(values)
        try:
            dbh = DBHandler('recommend')
            dbh.local_connect()
            results = dbh.select(sql, values)
        except Exception as e:
            logger.error('DB connect process error. {}'.format(e))
            return False
        else:
            logger.info('DB connect process ended normally.')
            return results
        finally:
            dbh.close()

            
    def recommend(self, item_id):
        """レコメンド商品情報取得"""
        
#         # レコメンド商品取得
#         r_sql = 'SELECT id, value FROM recommend WHERE id=%s'
#         r_items = self.db_select(r_sql, (item_id,))
#         # 返却値を商品IDのタプルに変換
#         r_item_id = tuple(ast.literal_eval(r_items[0][1])['id'])
        
        # predict
        result = kw_model.id_predict(item_id, topn=10)
        r_item_id = tuple([i[0] for i in result])

        # レコメンド商品の詳細情報を検索
        d_sql = generate_sql(table='item', idxs=ITEM_TABLE_IDX2, n_where=len(r_item_id))
        results = self.db_select(d_sql, (r_item_id))
        
        # 整形
        recommend_items = []
        for id_ in r_item_id:
            for item in results:
                if id_ == item[0]:
                    item_dict = {}
                    item_dict['id'] = item[0]
                    item_dict['name'] = item[1]
                    item_dict['name_jp'] = item[2]
                    item_dict['url'] = item[3]
                    p = str(item[4])
                    item_dict['price'] = p[:-3] + ',' + p[-3:] + '円'
                    item_dict['img_url'] = 'http://www.enoteca.co.jp/'+str(item[5])
                    item_dict['type2'] = item[6]
                    if item_dict['type2'] in ['スパークリング', 'スパークリングワイン', '赤スパークリング']:
                        item_dict['type_label'] = 'ext-sparkling'
                    elif item_dict['type2'] == '赤ワイン':
                        item_dict['type_label'] = 'ext-red'
                    elif item_dict['type2'] == '白ワイン':
                        item_dict['type_label'] = 'ext-white'
                    item_dict['desc'] = item[7]
                    recommend_items.append(item_dict)
        logger.debug('Recommend items = {}'.format(recommend_items))
        return recommend_items
    
        
class KeyWord(models.Model):
    
    def Search(self, word):
        result = kw_model.doc_predict(word, topn=10)
        r_item_id = tuple([i[0] for i in result])
        
        # レコメンド商品の詳細情報を検索
        d_sql = generate_sql(table='item', idxs=ITEM_TABLE_IDX2, n_where=len(r_item_id))
        results = self.db_select(d_sql, (r_item_id))
        
        # 整形
        recommend_items = []
        for id_ in r_item_id:
            for item in results:
                if id_ == item[0]:
                    item_dict = {}
                    item_dict['id'] = item[0]
                    item_dict['name'] = item[1]
                    item_dict['name_jp'] = item[2]
                    item_dict['url'] = item[3]
                    p = str(item[4])
                    item_dict['price'] = p[:-3] + ',' + p[-3:] + '円'
                    item_dict['img_url'] = 'http://www.enoteca.co.jp/'+str(item[5])
                    item_dict['type2'] = item[6]
                    if item_dict['type2'] in ['スパークリング', 'スパークリングワイン', '赤スパークリング']:
                        item_dict['type_label'] = 'ext-sparkling'
                    elif item_dict['type2'] == '赤ワイン':
                        item_dict['type_label'] = 'ext-red'
                    elif item_dict['type2'] == '白ワイン':
                        item_dict['type_label'] = 'ext-white'
                    item_dict['desc'] = item[7]
                    recommend_items.append(item_dict)
        logger.debug('Recommend items = {}'.format(recommend_items))
        return recommend_items
    
    
    def db_select(self, sql, values):
        """SELECT文実行"""
        # SQL実行処理
        logger.info('DB connect process start...')
        logger.debug(sql)
        logger.debug(values)
        try:
            dbh = DBHandler('recommend')
            dbh.local_connect()
            results = dbh.select(sql, values)
        except Exception as e:
            logger.error('DB connect process error. {}'.format(e))
            return False
        else:
            logger.info('DB connect process ended normally.')
            return results
        finally:
            dbh.close()
        
    