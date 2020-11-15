# ワイン検索＆レコメンデーションシステム

## 概要
ワイン画像をアップロードすると、そのワイン情報を画面に表示し、
同時にそのワインに類似したワインをレコメンドするアプリケーション。
レコメンドはワイン情報のテキストを使用した自然言語処理で実装。

※実際に実用化する場合はスマホアプリ化してカメラを起動できるようにすることを想定していますが、
　今回は通常のブラウザで実装し、画像アップロードする方法としています。

## 制作期間
2週間

## 使用した技術
- 画像検出
  - Efficientnet-b0のファインチューニング
- レコメンド
  - Doc2Vecによる自然言語処理（ワイン情報のテキスト文を学習）
 
## 主なフレームワーク・ライブラリ
- ディープラーニング
  - tensorflow
  - keras
  - gensim
- 形態素解析
  - Janome
- スクレイピング
  - BeautifulSoup4
- サーバサイド
  - Django


## 主な開発言語、環境
- 言語 : Python
- DB : MySQL
- クラウド : AWS

## 主なコード
- 自然言語処理レコメンド(Doc2Vec)
  - https://github.com/obata01/dic_project/blob/master/src/models/common/doc2vec.py 
- 形態素解析
  - https://github.com/obata01/dic_project/blob/master/src/models/common/tokenizer.py
- 画像分類(学習用)
  - https://github.com/obata01/dic_project/blob/master/src/recommend/img_detection_train.py
- DB操作用共通処理
  - https://github.com/obata01/dic_project/blob/master/src/db/dbhandler.py
- スクレイピング処理
  - (共通処理)https://github.com/obata01/dic_project/blob/master/src/scraping/simplescraping.py
  - (実行用)https://github.com/obata01/dic_project/blob/master/src/scraping/enoteca_scrape.py
- django関連
  - https://github.com/obata01/dic_project/tree/master/app

