import logging
import datetime

class Logger:
    def __init__(self, level):
        self.logger = logging.getLogger(__name__)
        fmt = "[{}] {} - {} ".format('%(levelname)s', '%(asctime)s', '%(message)s')

        level = level.lower()
        if level == 'info':
            logging.basicConfig(level=logging.INFO, format=fmt)
        elif level == 'debug':
            logging.basicConfig(level=logging.DEBUG, format=fmt)
        
    def info(self, msg):
        self.logger.info(msg)
        
    def error(self, msg):
        self.logger.error(msg)
        
    def warning(self, msg):
        self.logger.warning(msg)
        
    def debug(self, msg):
        self.logger.debug(msg)
    
    def now_string(self):
        return str(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))