#!/usr/bin/env python3

# 共通処理
# Webスクレイピング処理

import sys
import os
import re
import time
import pickle
import requests
from typing import Iterator
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from lib import utils
sys.setrecursionlimit(100000)

logger = utils.Logger(level='info')

TEMPORARY_ERROR_CODES = (408, 500, 502, 503, 504)  # 一時的なエラーを表すステータスコード。

class SimpleScraping:
    def __init__(self, sitemap_url, output_path):
        self.sitemap_url = sitemap_url
        self.output_path = output_path
        pass

        
    def get_urls(self, sitemap_url):
        """XML形式のサイトマップからURLを取得"""
        try:
            session = requests.Session()
            response = session.get(sitemap_url)
            soup = BeautifulSoup(response.text, 'xml')
            urls = soup.find_all('loc')
        except Exception as e:
            logger.error('Scraping sitemap is error. {}'.format(e))
        return urls
    
    
    def fetch(self, url: str, session) -> requests.Response:
        """
        指定したURLにリクエストを送り、Responseオブジェクトを返す。
        一時的なエラーが起きた場合は最大3回リトライする。
        3回リトライしても成功しなかった場合は例外Exceptionを発生させる。
        """
        max_retries = 3  # 最大で3回リトライする。
        retries = 0  # 現在のリトライ回数を示す変数。
        while True:
            try:
                logger.info('Retrieving {}'.format(url))
                response = session.get(url)
                if response.status_code not in TEMPORARY_ERROR_CODES:
                    return response  # 一時的なエラーでなければresponseを返して終了。

            except requests.exceptions.RequestException as ex:
                # ネットワークレベルのエラー（RequestException）の場合はログを出力してリトライする。
                logger.error('Network-level exception occured: {}'.format(ex))

            # リトライ処理
            retries += 1
            if retries >= max_retries:
                logger.error('Too many retries. So skip.')
                return response

            wait = 2**(retries - 1)  # 指数関数的なリトライ間隔を求める（**はべき乗を表す演算子）。
            logger.warning('Waiting {} seconds...'.format(wait))
            time.sleep(wait)  # ウェイトを取る。

            
    def scrape_detail_page(self, response: requests.Response) -> dict:
        """
        htmlパース処理
        """
        html_txt = response.text[:50000]   # 変数へ入れられるサイズに制限があるため足切り
        soup = BeautifulSoup(html_txt, 'html.parser')
        try:
            info = {
                'id': self.extract_key(response.url),
                'url': response.url,
                'title': soup.find('title').text
            }
        except Exception as e:
            return False  # 必要な情報が無くて処理をスキップして良い場合はFalseをreturn
        return info
    
            
    def run(self, urls, func_scrape):
        """
        クローリング＆スクレイピング実行処理
        """
        n = len(urls)
        session = requests.Session()
        for i in range(n):
            logger.info('{}/{} start...'.format(i+1, n))
            url = urls[i].text

            # html取得
            time.sleep(1)
            response = self.fetch(url, session)
            if 200 <= response.status_code < 300:
                logger.info('Request is Success. {} {}'.format(response, url))
            else:
                logger.error('Request is error. {} {}'.format(response, url))
                continue
                
            # htmlパース
            result = func_scrape(response)
            if result == False:
                logger.warning('URL:{} is skiped.'.format(url))
                continue
                
            # スクレイピング結果をファイルへ出力
            try:
                file_name = 'scraping_data_' + str(i) + '.pkl'
                file_name = os.path.join(self.output_path, file_name)
                self.write_file(result, file_name)
                logger.info('URL:{} is success.'.format(url))
            except Exception as e:
                logger.error('Unexpected write error. URL:{} {}'.format(url, e))
        del session, result
        gc.collect()
        logger.info('Run process is terminated normally.')
            
                
                
    def extract_key(self, url: str) -> str:
        """
        URLからキー（URLの末尾のISBN）を抜き出す
        """
        m = re.search(r'/([^/]+)$', url)  # 最後の/から文字列末尾までを正規表現で取得。
        return m.group(1)
                
    
    def write_file(self, data, out_file_name):
        """
        pickle形式でファイル出力する処理
        """
        with open(out_file_name, 'wb') as f:
            pickle.dump(data, f)

            
    def main(self):
        urls = self.get_urls(self.sitemap_url)
        self.run(urls, self.scrape_detail_page)

