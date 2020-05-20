# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import logging
import os
from logging.handlers import RotatingFileHandler

# logger settings
from config import DEBUG

LOG_PATH = os.getenv("LOG_FILE_PATH", "/var/log/")
os.makedirs(LOG_PATH, exist_ok=True)
log_file = os.path.join(LOG_PATH, '{}.log'.format("recommend"))

LOG_LEVEL = logging.DEBUG if DEBUG else logging.INFO
file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 10, backupCount=5)
file_handler.setLevel(LOG_LEVEL)
console = logging.StreamHandler()
console.setLevel(LOG_LEVEL)

logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)s] [%(funcName)s] %(message)s',
    handlers=[file_handler, console]
)
