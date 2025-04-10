import logging
from logging.handlers import RotatingFileHandler, SMTPHandler
import sys
from pythonjsonlogger import jsonlogger

class AiraLogger:
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self._configure_handlers()

    def _configure_handlers(self):
        # Configuración de formato JSON
        json_handler = RotatingFileHandler(
            'logs/aira.json',
            maxBytes=100*1024*1024,  # 100 MB
            backupCount=5,
            encoding='utf-8'
        )
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
        json_handler.setFormatter(json_formatter)

        # Configuración de consola
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)

        self.logger.addHandler(json_handler)
        self.logger.addHandler(console_handler)

    def log(self, level, message, **kwargs):
        self.logger.log(level, message, extra=kwargs)

    def debug(self, message, **kwargs):
        self.log(logging.DEBUG, message, **kwargs)

    def info(self, message, **kwargs):
        self.log(logging.INFO, message, **kwargs)

    def warning(self, message, **kwargs):
        self.log(logging.WARNING, message, **kwargs)

    def error(self, message, **kwargs):
        self.log(logging.ERROR, message, **kwargs)

    def critical(self, message, **kwargs):
        self.log(logging.CRITICAL, message, **kwargs)