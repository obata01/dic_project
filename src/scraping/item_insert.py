#!/usr/bin/env python3

# スクレイピングした商品情報を商品マスタとして
# DB(mysql)へインサートする処理

import mysql.connector
import os
import utils
import db
import pickle
import glob
import re
import sys
from time import sleep

logger = utils.Logger(level='DEBUG')

fin_id = []
DATA_DIR = '/home/ubuntu/my_project/data/raw/web'
    
if __name__ == '__main__':
    logger.info('Process start...')
    try:
        # ファイル取得
        all_files = glob.glob(DATA_DIR + '/*.log')
        n = len(all_files)

        # DB接続
        dbh = db.DBHandler('recommend')
        dbh.local_connect(cursor=True, prepared=True)
        stmt = "INSERT INTO item(id, name, name_jp, url, price, img_url_a, img_url_b, producer, \
        made_in, type1, type2, hinsyu, desc0, desc1, desc2, desc3, body) \
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);"
        
        for i, file in enumerate(all_files):
            logger.info('{}/{} | {} start.'.format(i+1, n, file))
            with open(file, 'rb') as f:
                data = pickle.load(f)
                item_id = data['id']
                if item_id in fin_id:
                    logger.info('{} {} is already inserted. So skip.'.format(file, item_id))
                    continue
                name = data['name']
                name_jp = data['name_jp']
                url = data['url']
                price = re.sub(r'[,円]', '', data['price'])
                img_url_a = re.sub(r'(.*\")(.*[jp][pn]g)(\".*)', '\\2', str(data['img_url_a']))
                img_url_b = re.sub(r'(.*\")(.*[jp][pn]g)(\".*)', '\\2', str(data['img_url_b']))
                producer = data['producer']
                made_in = data['made_in']
                type1 = data['type1']
                type2 = data['type2']
                hinsyu = data['hinsyu']
                desc0 = data['desc0']
                desc1 = data['desc1']
                desc2 = data['desc2']
                desc3 = data['desc3']
                body = data['body']

                hinsyu = re.sub(r'<td>', '', str(hinsyu))
                hinsyu = re.sub(r'</td>', '', str(hinsyu))
                hinsyu = re.sub(r'.*\">', '', str(hinsyu))
                hinsyu = re.sub(r'</.*', '||', str(hinsyu))
                hinsyu = re.sub('\n', '', str(hinsyu))
                
            columns = (item_id, name, name_jp, url, int(price), img_url_a, img_url_b, producer, 
                       made_in, type1, type2, hinsyu, desc0, desc1, desc2, desc3, str(body))
            dbh.pre_insert(stmt, columns)
            
            # インサート済みのIDを格納
            fin_id.append(item_id)
                            
    except Exception as e:
        logger.error('{} insert is something error. {}'.format(file, e))
        sys.exit(1)
    else:
        dbh.commit()
    finally:
        sleep(2)
        dbh.close()
        logger.info('Finish.')
        
