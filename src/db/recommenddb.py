#!/usr/bin/env python3

# DB名：recommend
# テーブル名・カラム名一覧情報格納用

import db

class RecommendDB(db.BaseDB):
    def __init__(self):
        self.__columns = {
            'item': {
                0: 'id', 1: 'name', 2: 'name_jp', 3: 'url', 4: 'price', 
                5: 'img_url_a', 6: 'img_url_b', 7: 'producer', 8: 'made_in', 9: 'type1', 
                10: 'type2', 11: 'hinsyu', 12: 'desc0', 13: 'desc1', 14: 'desc2', 
                15: 'desc3', 16: 'body'
            },
            'recommend': {
                0: 'id', 
                1: 'value'
            },
        }
        
    @property
    def columns(self):
        return self.__columns
