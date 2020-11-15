#!/usr/bin/env python3

# エノテカサイトからワイン画像をダウンロードする処理
# 画像URLはクロール＆スクレイピング済みの情報から使用する

import sys
import os
import pickle
import pprint
import time
import urllib.error
import urllib.request
import re

from download_images import DownloadImages as di
from lib import utils
logger = utils.Logger(level='info')


URL_HEADER = 'https://www.enoteca.co.jp'
PKL_FILE_DIR = '../../data/test/'
OUT_DIR = '../..'

def enoteca_get_img():
    """
    エノテカサイトから画像をダウンロードする処理
    """
    logger.info('Start get image precess...')
    fin_id = []

    n = 50
    for i in range(n):  # MAX:8323
        logger.info('{}/{} Start...'.format(i+1, n))
        file_name = 'scraping_data_' + str(i) + '.pkl'
        file_path = os.path.join(PKL_FILE_DIR, file_name)
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                id_ = data['id']
                img_url_a_ = str(data['img_url_a'])
                img_url_b_ = str(data['img_url_b'])
        except Exception as e1:
            logger.warning('file load error.{}'.format(e1))
            continue

        if 'http' not in img_url_a_:
            try:
                img_url_a_ = re.sub('(.*\")(.*)(\".*)', '\\2', img_url_a_)
                img_url_a_full = URL_HEADER + img_url_a_
            except Exception as e:
                logging.warning('url:{} can not join url head. {}'.format(img_url_a_, e))

        if 'http' not in img_url_b_:
            try:
                img_url_b_ = re.sub('(.*\")(.*)(\".*)', '\\2', img_url_b_)
                img_url_b_full = URL_HEADER + img_url_b_
            except Exception as e:
                logging.warning('url:{} can not join url head.'.format(img_url_b_, e))

        if id_ in fin_id:
            logging.warning('id:{} already exists. So the process is skipped. {} & {}'.format(id_, img_url_a_, img_url_b_))
            continue
            
        time.sleep(1)
        di.download(img_url_a_full, OUT_DIR+img_url_a_)
        time.sleep(1)
        di.download(img_url_b_full, OUT_DIR+img_url_b_)
        fin_id.append(id_)
        

enoteca_get_img()