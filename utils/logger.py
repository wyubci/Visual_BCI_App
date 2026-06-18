import logging
import os
import io
from loguru import logger


class GetLogging:
    """
    日志配置
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self):
        self.str_io = io.StringIO()
        # 错误日志
        logger.add(
            os.path.join("logs/ERROR/{time:YYYY-MM-DD}.log"),
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            filter=lambda x: True if x["level"].name == "ERROR" else False,
            rotation="00:00", retention=7, level='ERROR', encoding='utf-8'
        )
        # 警告日志
        logger.add(
            os.path.join("logs/WARNING/{time:YYYY-MM-DD}.log"),
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            filter=lambda x: True if x["level"].name == "WARNING" else False,
            rotation="00:00", retention=7, level='WARNING', encoding='utf-8',
        )
        # 普通记录日志
        logger.add(
            os.path.join("logs/INFO/{time:YYYY-MM-DD}.log"),
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            filter=lambda x: True if x["level"].name == "INFO" else False,
            rotation="00:00", retention=7, level='INFO', encoding='utf-8',
        )

        # debug日志
        logger.add(
            sink=logging.StreamHandler(self.str_io),
            format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
            # rotation="00:00", retention=7, level='DEBUG', encoding='utf-8'
        )

        self.logger = logger

    def debug(self, text):
        self.logger.debug(text)

    def info(self, text):
        self.logger.info(text)

    def error(self, text):
        self.logger.error(text)

    def warning(self, text):
        self.logger.warning(text)

    def show_log(self):
        self.str_io.seek(0)
        self.str_io.truncate()

    @classmethod
    def get_logger(cls):
        return cls()

logger = GetLogging().get_logger()

if __name__ == '__main__':
    l = GetLogging().get_logger()
    l.warning('aaa')