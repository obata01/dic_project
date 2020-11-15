#!/usr/bin/env python3

# DB接続情報

import os

class RdsConfig:
    def __init__(self, db_name):
        self.__endpoint = 'man-database-01.cluster-cuvaun7oz5f3.us-east-1.rds.amazonaws.com'
        self.__port = '3306'
        self.__user = '**********'
        self.__region = 'us-east-1'
        self.__dbname = db_name
        os.environ['LIBMYSQL_ENABLE_CLEARTEXT_PLUGIN'] = '1'
     
    @property
    def endpoint(self):
        return self.__endpoint
    
    @property
    def port(self):
        return self.__port
    
    @property
    def user(self):
        return self.__user
    
    @property
    def region(self):
        return self.__region
    
    @property
    def dbname(self):
        return self.__dbname

    
class LocalRdsConfig:
    def __init__(self, db_name):
        self.__port = '3306'
        self.__user = '**********'
        self.__passwd = '**********'
        self.__host = 'localhost'
        self.__dbname = db_name
    
    @property
    def port(self):
        return self.__port
    
    @property
    def user(self):
        return self.__user
    
    @property
    def passwd(self):
        return self.__passwd
    
    @property
    def host(self):
        return self.__host
    
    @property
    def dbname(self):
        return self.__dbname

