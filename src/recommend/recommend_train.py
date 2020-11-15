#!/usr/bin/env python3

# ワイン説明文レコメンド学習用

from recommend import ContentsRecommend

if __name__ == '__main__':
    cr = ContentsRecommend(model_load=False)
    cr.fit(epochs=1)