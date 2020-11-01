import mysql.connector
import boto3
import sys
import os

sys.path.append(os.path.abspath(".."))
from conf import dbconf
import utils

logger = utils.Logger(level='info')

class DBHandler:
    def __init__(self, db_name):
        self.db_name = db_name
        self.session = None
        self.client = None
        self.token = None
        self.cur = None
        self.conn = None
    
    def connect(self):
        conf = dbconf.RdsConfig(self.db_name)
        try:
            self.session = boto3.Session(profile_name='default')
            self.client = boto3.client('rds', region_name=conf.region)
            self.token = client.generate_db_auth_token(DBHostname=conf.endpoint, Port=conf.port, DBUsername=conf.user, Region=conf.region)
            self.conn =  mysql.connector.connect(host=conf.endpoint, user=conf.user, passwd=self.token, port=conf.port, database=conf.dbname)
            self.cur = conn.cursor(prepared=True)
#             self.cur = conn.cursor(prepared=True)
        except Exception as e:
            logger.error('DB connection error.'.format(e))
        else:
            logger.info('DB connection is terminate normally.')


    def local_connect(self, cursor=False, prepared=False):
        conf = dbconf.LocalRdsConfig(self.db_name)
        try:
            self.conn = mysql.connector.connect(
                     host=conf.host,
                     port=conf.port,
                     db=conf.dbname,
                     user=conf.user,
                     password=conf.passwd,
                     charset='utf8')
            if cursor:
                self.cur = self.conn.cursor(prepared=prepared)
        except Exception as e:
            logger.error('DB connection error.'.format(e))
        else:
            logger.info('DB connection is terminate normally.')
    
    
    def close(self):
        logger.info('DB connection release start...')
        try:
            self.conn.close()
            self.cur.close()
            self.__init__(self.db_name)
        except Exception as e:
            logger.error('DB connection release failed... {}'.format(e))
        else:
            logger.info('DB connection released successfully.')
            
        
    def is_close(self):
        """接続がクローズされているかをチェックする処理"""
        pass
    
    
    def commit(self):
        logger.info('Commit process start...')
        try:
            self.conn.commit()
        except Exception as e:
            logger.error('Commit process error. {}'.format(e))
        else:
            logger.info('Commit process ended normally.')
    
    
    def rollback(self):
        logger.info('Rollback process start...')
        try:
            self.conn.rollback()
        except Exception as e:
            logger.error('Rollback process error.{}'.format(e))
        else:
            logger.info('Rollback process ended normally.')
    
    
    def select(self, sql, ph_values=None):
        logger.info('SQL execute start...')
        try:
            if self.conn == None:
                return False
            self.cur = self.conn.cursor(buffered=True)
            if ph_values == None:
                self.cur.execute(sql)
            else:
                self.cur.execute(sql, ph_values)
            results = self.cur.fetchall()
        except Exception as e:
            logger.error('SQL error. {}'.format(e))
            return False
        else:
            logger.info('SQL ended normally.')
            return results
        
        
    def pre_insert(self, stmt, values):
        logger.info('Insert process start...')
        for i, v in enumerate(values):
            logger.info('{} {}'.format(i, v))
        try:
            if self.conn == None:
                return False
            self.cur.execute(stmt, values)
        except Exception as e:
            logger.error('insert process error. {}'.format(e))
            self.rollback()
            sys.exit(1)
        else:
            logger.info('Insert process ended normally.')
            return True
        