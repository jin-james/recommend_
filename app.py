import os
import time
import logging
import requests
from keras import models

import config
import tensorflow as tf
from config import CONNECT_TIMEOUT, READ_TIMEOUT
from flask import Flask
from utils.kerasbert_classify import question_classify
from config import DATA_TRANS_PORT, DATA_TRANS_HOST, DEFAULT_CLIENT_ID, DEFAULT_CLIENT_SECRET

logger = logging.getLogger(__name__)

app = Flask(__name__)

global graph
graph = tf.compat.v1.get_default_graph()
model = models.load_model('./keras_classify_model/keras_bert_lishi_cz.h5')
app.config.from_object(config)
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
    "Content-Type": "application/json",
    'Connection': 'close',
    'Authorization': 'Potent {} {}'.format(DEFAULT_CLIENT_ID, DEFAULT_CLIENT_SECRET)
}


# api.init_app(app)

@app.route('/recommend/knowledge/<string:question_id>', methods=['GET'])
def get_know(question_id):
    # question_data = data_trans.get_file_data(question_id)
    url = "http://{}:{}/yangtze/recommend_ques/info/{}".format(DATA_TRANS_HOST, DATA_TRANS_PORT, question_id)
    t1 = time.time()

    resp = requests.get(url=url, params=None, headers=headers, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
    question_data = resp.json()
    question_data = question_data["data"]
    with graph.as_default():
        point_str = question_classify(question_data, headers, model=model)
    data = {
        "data": point_str
    }
    t2 = time.time()
    print(t2-t1)
    return data


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
    # app.run()
