import abc

class BaseDB(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        self.__columns = {}
        pass
    
    @abc.abstractmethod
    def columns(self, table_name):
        pass