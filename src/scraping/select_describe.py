#!/home/ubuntu/anaconda3/envs/tensorflow_p36/bin/python3

# MySQLのitemテーブルから商品説明文を抽出して
# 指定ディレクトリに "item_id.txt"の形式で出力する処理。

import mysql.connector
import os
import utils
import db
import pickle
import glob
import re
from db import DBHandler
from db import RecommendDB

logger = utils.Logger(level='INFO')

DATA_DIR = '/home/ubuntu/my_project/src/data/wine_desc3_ajiwai'


def db_select(sql):
    """SELECT文実行"""
    logger.info('DB connect process start...')
    logger.info(sql)
    try:
        dbh = DBHandler('recommend')
        dbh.local_connect()
        results = dbh.select(sql)
    except Exception as e:
        logger.error('DB connect process error. {}'.format(e))
        return False
    else:
        logger.info('DB connect process ended normally.')
        return results
    finally:
        dbh.close()


        
def file_write(results):
    """ファイル出力"""
    logger.info('File writing process start...')
    try:
        for item_id, value0, value1, value2 in results:
            value = value0 + value1 + value2
            if len(value) < 5:
                logger.warning('{} value length is too short. So skip.'.format(item_id))
                continue
            output_path = os.path.join(DATA_DIR, item_id+'.txt')
            with open(output_path, mode='w') as f:
                f.write(value)
                logger.info('{} finish.'.format(output_path))
    except Exception as e:
        logger.error('{} file writing error. {}'.format(output_path, e))
        return False
    else:
        logger.info('File writing process ended normally.')
        return True
    
    
     
if __name__ == '__main__':
    
    # 初期化・変数設定
    rdb = RecommendDB()
    item = rdb.columns['item']
    item_id = item[0]
    desc0 = item[12]
    desc1 = item[13]
    desc3 = item[15]
    sql = "SELECT " +item_id+", "+desc0+", "+desc1+", "+desc3+" FROM item"
    
    # SELECT文実行
    results = db_select(sql)
    if not results:
        raise Exception('Error.')
        sys.exit(1)
    
    # ファイル書き出し
    r = file_write(results)
    if not r:
        raise Exception('Error.')
        sys.exit(1)
    
    
