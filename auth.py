# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from flask import g
from flask_httpauth import HTTPTokenAuth

from config import TOKEN_SCHEME
from utils.constants import FIXED_TOKEN

token_auth = HTTPTokenAuth(scheme=TOKEN_SCHEME)


@token_auth.verify_token
def verify_token(token):
    if not token:
        return False
    if token == FIXED_TOKEN:
        g.user = {
            "is_admin": 0,
            "name": "guest",
            "gender": "U",
            "status": "0",
            "schools": []
        }
        return True
    return False
