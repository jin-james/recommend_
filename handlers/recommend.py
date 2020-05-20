# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from flask import request
from flask_restful import Resource, marshal_with

from base import resource_fields, APIResponse
from utils.logutils import logging
from utils.data_trans import data_trans
from utils.response import PARAMS_ERROR
from utils.kerasbert_classify import question_classify
from auth import token_auth

logger = logging.getLogger(__name__)


class QuesMapKnowledgeEndpoint(Resource):
    decorators = [marshal_with(resource_fields), token_auth.login_required]

    def get(self, question_id):
        """
        @api {GET} /recommend/knowledge/ 上传文件

        @apiGroup Common

        @apiExample 返回值
        {

        }
        """
        params = request.form.to_dict()
        if not question_id:
            return APIResponse(code=PARAMS_ERROR, message="题目ID参数错误")
        question_data = data_trans.get_file_data(question_id)
        point_str = question_classify(question_data)
        data = {
            "data": point_str
        }
        return APIResponse(data=data)
