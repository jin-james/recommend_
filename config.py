# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import os

DEBUG = str(os.getenv("DEBUG", True)).lower() in ["true", "yes", "y", "on", "1"]

TOKEN_SCHEME = "Token"
SECRET_KEY = os.environ.get("SECRET_KEY", "U2FsdGVkX19LyQRMPTdjRkyc4K4Txjy/QeYpbyBjWyI=")
EXPIRATION = int(os.environ.get("EXPIRATION", 12 * 60 * 60))

FILE_URL = os.getenv("FILE_URL", "/data/office-resource")

# DATA_TRANS_HOST = os.getenv("DATA_TRANS_HOST", "172.25.15.78")
DATA_TRANS_HOST = os.getenv("DATA_TRANS_HOST", "106.54.128.176")
DATA_TRANS_PORT = os.getenv("DATA_TRANS_PORT", "8009")

BERT_IP = os.getenv("BERT_IP", "10.88.190.154")

# 虚拟用户
DEFAULT_CLIENT_ID = "RRvVOGmcYldZSSDF"
DEFAULT_CLIENT_SECRET = "RrsWRvVOXmzRmRvdZSSHGmcYlWMgbDzW"

CONNECT_TIMEOUT = os.getenv("CONNECT_TIMEOUT", 3)
READ_TIMEOUT = os.getenv("READ_TIMEOUT", 10)
