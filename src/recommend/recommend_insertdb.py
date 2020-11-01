import mysql.connector
import os
import utils
import db
import pickle
import glob
import re
import sys
import gc
from recommend import ContentsRecommend

logger = utils.Logger(level='INFO')

def select_itemid(sql):
    """SELECT文実行"""
    logger.info('DB connect process start...')
    logger.info(sql)
    try:
        dbh = db.DBHandler('recommend')
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
        del dbh
        gc.collect()


    
if __name__ == '__main__':
    logger.info('Process start...')
    try:
        # SQL(select)
        sql = "SELECT id from item"
        id_list = select_itemid(sql)
        n = len(id_list)
        logger.info(id_list)
        
        # DB接続
        dbh = db.DBHandler('recommend')
        dbh.local_connect(cursor=True, prepared=True)
        stmt = "INSERT INTO recommend(id, value) VALUES (?, ?);"
        
        # ContentsRecommend 初期化
        cr = ContentsRecommend(model_load=True)
        
        for i, item_id in enumerate(id_list):
            item_id = str(item_id[0])
            logger.info('{}/{} | {} start.'.format(i+1, n, item_id))
            
            # predict
            results = cr.id_predict(item_id, topn=10)
            if results == None or len(results) == 0:
                logger.warning('{} is skip. Recommend item is none.'.format(item_id))
                continue
            val_dict = {}
            ids = []
            sims = []
            for id_, sim in results:
                ids.append(id_)
                sims.append(sim)                
            val_dict['id'] = ids
            val_dict['sim'] = sims
                 
            # insert実行
            dbh.pre_insert(stmt, (item_id, str(val_dict)))
            
    except Exception as e:
        logger.error('{} insert is something error. {}'.format(file, e))
        sys.exit(1)
    else:
        dbh.commit()
    finally:
        dbh.close()
        logger.info('Finish.')
        
