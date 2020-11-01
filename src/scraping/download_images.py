import sys
import os
import urllib.error
import urllib.request
import utils
logger = utils.Logger(level='info')


class DownloadImages:
    def __init__(self):
        pass
    
    @classmethod
    def download(self, url, out_path):
        """
        引数urlから画像をDLしてout_pathへ書き出す処理
        """
        logger.info(sys._getframe().f_code.co_name + ' - Process start...')
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            with opener.open(url) as web_file:
                data = web_file.read()
                with open(out_path, mode='wb') as local_file:
                    local_file.write(data)
                    logger.info('url:{} is success.'.format(url))
        except Exception as e:
            logger.error('{} - Process error. {}'.format(sys._getframe().f_code.co_name, e))
        else:
            logger.info('{} - Prpcess ended normally.'.format(sys._getframe().f_code.co_name))
            return labels