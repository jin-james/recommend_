# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from flask_restful import fields

from utils.response import MESSAGE

resource_fields = {
    "code": fields.Integer,
    "message": fields.String,
    "data": fields.Raw(default=None)
}


class APIResponse(object):

    def __init__(self, code=0, message="success", data=None):
        if not message:
            message = MESSAGE.get(code)
        self.code = code
        self.message = message
        self.data = data

    def to_json(self):
        return {
            "code": self.code,
            "message": self.message,
            "data": self.data
        }
