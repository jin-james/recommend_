# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

from flask_restful import Api
from handlers.recommend import QuesMapKnowledgeEndpoint


api = Api()

api.add_resource(QuesMapKnowledgeEndpoint, "/recommend/knowledge", strict_slashes=False)
