import mysql.connector
import os
import utils
import db
import pickle
import glob
import re
import mysql.connector
from db import DBHandler
from db import RecommendDB

logger = utils.Logger(level='INFO')


class ItemInfoSelect():
    def __init__(self):
        pass

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