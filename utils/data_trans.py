import logging
import uuid

from winney import Winney
import config

logger = logging.getLogger(__name__)


class DataTrans(object):

    def __init__(self, host, port):
        self.winney = Winney(host=host, port=port, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
            "Content-Type": "application/json",
            'Connection': 'close',
            'Authorization': '{}'.format('Token e4d7c259ca2f23f80d269a2f5c8c835443c52991')
        })
        self.init_functions()

    def init_functions(self):
        self.winney.add_url(method="get", uri="/question/{uid}/", function_name="get_file_data")
        self.winney.add_url(method="get", uri="/knowledge/tree/", function_name="get_tree")

    @staticmethod
    def get_data(r):
        if not r.ok():
            return None
        data = r.get_json()
        if data["code"] != 0:
            print("Failed to request, response = ", data)
            return None
        return data["data"]

    def get_file_data(self, uid):
        r = self.winney.get_file_data(uid=uid)
        return self.get_data(r)

    def get_tree(self, subject_id):
        r = self.winney.get_tree(data={"subject_id": subject_id, "name": "", "attach": 0})
        return r.json()


data_trans = DataTrans(config.DATA_TRANS_HOST, config.DATA_TRANS_PORT)
