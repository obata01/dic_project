#!/usr/bin/env python3

# DB名毎にテーブル名・カラム名を保持するための
# 抽象基底クラス

import abc

class BaseDB(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        self.__columns = {}
        pass
    
    @abc.abstractmethod
    def columns(self, table_name):
        pass