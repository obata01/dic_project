#!/usr/bin/env python3

# エノテカサイトをクロール＆スクレイピングして
# ワイン情報を取得する処理


from simplescraping import *

TYPE_LIST = ['ワイン']
TEMPORARY_ERROR_CODES = (408, 500, 502, 503, 504)  # 一時的なエラーを表すステータスコード。
SITEMAP_URL = 'https://www.enoteca.co.jp/sitemap_detail.xml'
fin_list = []

class EnotecaScraping(SimpleScraping):
    def __init__(self, sitemap_url, output_path):
        super().__init__(sitemap_url, output_path)
    
    
    def scrape_detail_page(self, response: requests.Response) -> dict:
        item_id = self.extract_key(response.url)
        if item_id in fin_list:
            return False

        html_txt = response.text[:50000]
        soup = BeautifulSoup(html_txt, 'html.parser')


        # ヴィンテージ情報と味わいの取得
        desc = soup.find_all('p', class_='cy-stock__text')
        desc_1, desc_2 = 0, 0
        if len(desc) == 1:
            desc_2 = desc[0].text
        elif len(desc) >= 2:
            desc_1 = desc[0].text
            desc_2 = desc[1].text


        # 生産者・生産地・タイプ・品種の取得
        table = soup.find('table', class_='cy-product__table')
        for i, val in enumerate(table.find_all('th')):
            val = val.text
            if val == '生産者':
                producer_idx = i
            elif val == '生産地':
                made_in_idx = i
            elif val == 'タイプ':
                type_idx = i
            elif val == '品種':
                hinsyu_idx = i

        try:
            producer = table.find_all('td')[producer_idx].find('a').text
        except:
            producer = 0
            logger.info('producer is not exist.')
        try:
            made_in = table.find_all('td')[made_in_idx].find('a').text
        except:
            made_in = 0
            logger.info('made_in is not exist.')
        try:
            type1 = table.find_all('td')[type_idx].find('a').text
        except:
            type1 = 0
            logger.info('type1 is not exist.')
        try:
            type2 = table.find_all('td')[type_idx].find_all('a')[1].text
        except:
            type2 = 0
            logger.info('type2 is not exist.')
        try:
            hinsyu = table.find_all('td')[hinsyu_idx]
        except:
            hinsyu = 0
            logger.info('hinsyu is not exist.')

        if type1 not in TYPE_LIST:
            logger.info('this item is not in type list.')
            return False

        body = soup.find('p', class_='cy-product__body__image')
        if body != None:
            body = body.find('img')


        info = {
            'id': item_id,
            'url': response.url,
            'name': soup.find('div', class_='cy-prodcut__ttl').find('p').text,
            'name_jp': soup.find('div', class_='cy-prodcut__ttl').find('h1').text,
            'price': soup.find('p', class_='cy-product__price').find('strong').text,
            'img_url_a': soup.find('a', class_='js-zoom_open').find('img',src=re.compile('^/item/img/')),
            'img_url_b': soup.find_all('a', class_='js-zoom_open')[1].find('img',src=re.compile('^/item/img/')),
            'producer': producer,
            'made_in': made_in,
            'type1': type1,
            'type2': type2,
            'hinsyu': hinsyu,
            'desc0': soup.find('p', class_='cy-product__lead').text,
            'desc1': soup.find('p', class_='cy-product__text').text,
            'desc2': desc_1,
            'desc3': desc_2,
            'body': body,
        }
        fin_list.append(item_id)
        return info

if __name__ == '__main__':
    ss = EnotecaScraping(SITEMAP_URL, OUTPUT_PATH)
    ss.main()