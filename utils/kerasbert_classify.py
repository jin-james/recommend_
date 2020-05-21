import json
import re
import time
import uuid
from collections import defaultdict  # 用于创建一个空的字典，在后续统计词频可清理频率少的词语

import jieba
import jieba.analyse
import numpy as np
import requests

from app import READ_TIMEOUT, CONNECT_TIMEOUT
from config import BERT_IP, DATA_TRANS_HOST, DATA_TRANS_PORT, DEFAULT_CLIENT_ID, DEFAULT_CLIENT_SECRET
import yaml
from keras.backend import clear_session
from bert_serving.client import BertClient
from gensim import corpora, similarities, models as g_models
from keras import losses, Sequential, models
from keras.layers import Dense, Bidirectional, LSTM
from keras.optimizers import Adam
# from keras.preprocessing import sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


CHOICE_QUESTIONS = ["选择题", "单选题", "单项选择题", "多选题", "多项选择题", "听力选择题", '双选题']  # 选择题
maxlen = 100  # 句子的最大长度，padding要用的


def get_token_dict(dict_path):
    '''
    :param: dict_path: 是bert模型的vocab.txt文件
    :return:将文件中字进行编码
    '''
    # 将bert模型中的 字 进行编码
    # 目的是 喂入模型  的是  这些编码，不是汉字
    token_dict = {}
    with open(dict_path, encoding='utf-8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


# class OurTokenizer(Tokenizer):
#     '''
#     扩充 vocab.txt文件的
#     '''
#
#     def _tokenize(self, text):
#         R = []
#         for c in text:
#             if c in self._token_dict:
#                 R.append(c)
#             elif self._is_space(c):
#                 R.append('[unused1]')
#             else:
#                 R.append('[UNK]')
#         return R


def get_data(one_dict):
    '''
    读取数据的函数
    :return: list  类型的 数据
    '''
    # pos = []
    # neg = []
    # with codecs.open('./data/pos.txt', 'r', 'utf-8') as reader:
    # 	for line in reader:
    # 		pos.append(line.strip())
    # with codecs.open('./data/neg.txt', 'r', 'utf-8') as reader:
    # 	for line in reader:
    # 		neg.append(line.strip())
    # return pos, neg
    length = []
    contents = []
    for key, value in one_dict.items():
        leg = len(value)
        if leg > 50:
            length.append(leg)
            for con in value:
                contents.append(con)
    return length, contents


# 得到编码
# def get_encode(contents, token_dict):
#     '''
#     :param pos:第一类文本数据
#     :param neg:第二类文本数据
#     :param token_dict:编码字典
#     :return:[X1,X2]，其中X1是经过编码后的集合，X2表示第一句和第二句的位置，记录的是位置信息
#     '''
#     all_data = contents
#     tokenizer = OurTokenizer(token_dict)
#     X1 = []
#     X2 = []
#     for line in all_data:
#         # tokenizer.encode(first,second, maxlen)
#         # 第一句和第二句，最大的长度，
#         # 本数据集是  都是按照第一句，即一行数据即是一句，也就是第一句
#         # 返回的x1,是经过编码过后得到，纯整数集合
#         # 返回的x2,源码中segment_ids，表示区分第一句和第二句的位置。结果为：[0]*first_len+[1]*sencond_len
#         # 本数据集中，全是以字来分割的。
#         # line_list = line.split('.')
#         # for i in line_list:
#         #     print(i)
#         # line = line.encode('gbk')
#         x1, x2 = tokenizer.encode(first=line)
#         X1.append(x1)
#         X2.append(x2)
#     # 利用Keras API进行对数据集  补齐  操作。
#     # 与word2vec没什么区别
#     X1 = sequence.pad_sequences(X1, maxlen=maxlen, padding='post', truncating='post')
#     X2 = sequence.pad_sequences(X2, maxlen=maxlen, padding='post', truncating='post')
#     return [X1, X2]


# def build_bert_model(X1, X2):
#     '''
#     :param X1:经过编码过后的集合
#     :param X2:经过编码过后的位置集合
#     :return:模型
#     '''
#
#     bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
#     # config_path 是Bert模型的参数，checkpoint_path 是Bert模型的最新点，即训练的最新结果
#     # 注：https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip，
#     #     下载完之后，解压得到4个文件，直接放到 项目的路径下，要写上绝对路径，以防出现问题。
#     wordvec = bert_model.predict([X1, X2])
#     return wordvec


def build_model(label_count):
    model = Sequential()
    model.add(Bidirectional(LSTM(256), input_shape=(1, 768)))
    model.add(Dense(label_count, activation='softmax'))
    model.compile(loss=losses.categorical_crossentropy, optimizer=Adam(1e-5), metrics=['accuracy'])

    return model


def train(X_train, Y_train, label_count):
    t1 = int(round(time.time() * 1000))
    model = build_model(label_count)
    model.fit(X_train, Y_train, batch_size=32, epochs=200, validation_split=0.2)
    model.summary()
    # yaml_string = model.to_yaml()
    # with open('./keras_classify_model/keras_bert{}.yml'.format(t1), 'w') as f:
    #     f.write(yaml.dump(yaml_string, default_flow_style=True))
    # model.save_weights('./keras_classify_model/keras_bert{}.h5'.format(t1))
    model.save('./keras_classify_model/keras_bert{}.h5'.format(t1))


def stop_words_list():
    stopwords = [line.strip() for line in open('./static/stop_words.txt', 'r').readlines()]
    return stopwords


def get_similarity(ques_content, string):
    data = []
    for str in string:
        data1 = jieba.cut(str)
        # core_word = jiagu.keywords(str, 5)
        data.append(data1)
    # data.append(core_word)
    for i in range(len(data)):
        data2 = ""
        for j in data[i]:
            data2 += j + " "
        data[i] = data2
    documents = [lis for lis in data]
    texts = [[word for word in document.split()] for document in documents]
    # 计算词语的频率
    frequency = defaultdict(int)
    for text in texts:
        for word in text:
            frequency[word] += 1
    # 对频率低的词语进行过滤（可选）
    # texts = [[word for word in text if frequency[word] > 3] for text in texts]

    # 通过语料库将文档的词语进行建立词典
    dictionary = corpora.Dictionary(texts)
    # 加载要对比的文档
    data3 = jieba.cut(ques_content)
    data31 = ""
    for i in data3:
        data31 += i + " "
    # 将要对比的文档通过doc2bow转化为稀疏向量
    new_xs = dictionary.doc2bow(data31.split())
    # 对语料库进一步处理，得到新语料库
    corpus = [dictionary.doc2bow(text) for text in texts]
    # 将新语料库通过tf-idf model 进行处理，得到tfidf
    tfidf = g_models.TfidfModel(corpus)
    # 通过token2id得到特征数
    featurenum = len(dictionary.token2id.keys())
    # 稀疏矩阵相似度，从而建立索引
    index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=featurenum)
    # 得到最终相似结果
    similarity = index[tfidf[new_xs]]
    return similarity


def get_point_list(know_label, simi_score):
    point_list = {}
    for j in range(len(know_label)):
        point_list[know_label[j]] = simi_score[j]
    return point_list


def get_point_list_(know_label, simi_score):
    point_list = []
    count = 0
    for i in range(len(know_label)):
        list = know_label[i]
        # score_list = sim_score[i]
        for j in range(len(list)):
            point_list.append({
                list[j]: simi_score[count]
            })
            count += 1
    if count < len(simi_score):
        point_list.append({
            know_label[-1][-1]: simi_score[count]
        })
    return point_list


def sort(point_list):
    return_point = {}
    for point in point_list:
        for key, value in point.items():
            return_point[key] = value
    points = sorted(return_point.items(), key=lambda k: k[1], reverse=True)
    return points


def get_return(ques_seg, points_tfidf):
    points_str = []
    word4, word2 = [], []
    for word in ques_seg:
        if len(word) >= 3:
            word4.append(word)
        else:
            word2.append(word)
    for item in points_tfidf:
        for word in word4:
            if word in item[0] and item[0] not in points_str:
                points_str.append(item[0])
    for item in points_tfidf:
        for word in word2:
            if word in item[0] and item[0] not in points_str:
                points_str.append(item[0])
    if len(points_str) > 3:
        points_str = points_str[0:3]
    else:
        for item in points_tfidf:
            if item[0] not in points_str and len(points_str) < 3:
                points_str.append(item[0])
    return points_str


def get_return_(ques_seg, points_tfidf):
    points_str = []
    word4, word2 = [], []
    for word in ques_seg:
        if len(word) >= 3:
            word4.append(word)
        else:
            word2.append(word)
    for item in points_tfidf:
        for word in word4:
            if word in item and item not in points_str:
                points_str.append(item)
    for item in points_tfidf:
        for word in word2:
            if word in item and item not in points_str:
                points_str.append(item)
    if len(points_str) > 3:
        points_str = points_str[0:3]
    else:
        for item in points_tfidf:
            if item not in points_str and len(points_str) < 3:
                points_str.append(item)
    return points_str


def per_knowledge(json_path):
    label = []
    with open(json_path, 'r', encoding='utf-8-sig') as load_f:
        load_list = json.load(load_f)
        for per in load_list:
            per_label = []
            children = per.get('children')
            label_str = per.get('label').encode('gbk').decode('gbk')
            recurrent(children, label_str, per_label)
            label.append(per_label)
    return label


def recurrent(children, label_str, label):
    if not children:
        label.append(label_str)
    else:
        label.append(label_str)
        for child in children:
            sub_children = child.get('children')
            label_str = child.get('label').encode('gbk').decode('gbk')
            recurrent(sub_children, label_str, label)


def per_knowledge_(json_data):
    label = []
    for per in json_data:
        per_label = []
        children = per.get('subs')
        label_str = per.get('name')
        recurrent_(children, label_str, per_label)
        label.append(per_label)
    return label


def recurrent_(children, label_str, label):
    if not children:
        label.append(label_str)
    else:
        label.append(label_str)
        for child in children:
            sub_children = child.get('subs')
            label_str = child.get('name')
            recurrent_(sub_children, label_str, label)


def pre_test(ques_path, json_path=None, know_label=None):
    if not know_label and json_path:
        know_label = per_knowledge(json_path)  # 知识点列表
    else:
        know_label = know_label
    one = {}
    with open(ques_path, 'r', encoding='utf-8') as load_f:
        load_list = json.load(load_f)
        data = load_list.get('data')
        contents, knows = [], []
        for item in data:
            content = item.get('question').replace('&nbsp;', '').replace('<p>', '').replace('</p>', '')
            content = re.sub(r'（.*?）', "", content)
            answer = item.get("answer").replace('&nbsp;', '').replace('<p>', '').replace('</p>', '')
            options = item.get("option")
            q_type_name = item.get("type")
            know = item.get("knowledge")
            if q_type_name in CHOICE_QUESTIONS:
                for sub_item in options:
                    if answer == sub_item.get("key"):
                        content += sub_item.get("value")
            else:
                content += answer
            for k in range(len(know_label)):
                k_list = know_label[k]
                kn = know.split(' ')
                for kk in kn:
                    if kk in k_list:
                        key = k
                        if not one.get(key):
                            one[key] = []
                        one[key].append(content)
                        break
            contents.append(content)
            knows.append(know)
    return contents, knows, know_label, one


def get_tree_data(subject_id, headers):
    url = "http://{}:{}/yangtze/recommend/tree/{}".format(DATA_TRANS_HOST, DATA_TRANS_PORT, subject_id)
    resp = requests.get(url=url, params=None, headers=headers)
    json_data = resp.json()
    return json_data


def question_classify(item, headers, model=None):
    """
    :param kwards: 包含学校和用户信息
    :param item: 一道题的所有信息
    :return:
    """

    # con, knows, know_label_, one_dict = pre_test()
    if item:
        subject = item.get("subject")
        subject_name = subject.get("name")
        subject_id = subject.get("uid")
        phase = subject.get("phase")
        phase_name = phase.get("name")
        content = item.get('content').replace('&nbsp;', '').replace('<p>', '').replace('</p>', '')
        content = re.sub(r'（.*?）', "", content)
        answer = item.get("answer").replace('&nbsp;', '').replace('<p>', '').replace('</p>', '')
        options = item.get("options")
        q_type_name = item.get("q_type").get("name")
        if q_type_name in CHOICE_QUESTIONS:
            for sub_item in options:
                if answer == sub_item.get("key"):
                    content += sub_item.get("content") + "。"
        else:
            content += answer + "。"
        if subject_name == "历史" and phase_name == "初中":
            json_path = r'./json_data/zx-knowledge-cz-ls.json'
            know_label_ = per_knowledge(json_path)
            content = [content]

            # token_dict = get_token_dict(dict_path)
            # [X1, X2] = get_encode(content, token_dict)
            # wordvec = build_bert_model(X1, X2)
            # bc = BertClient(ip='118.24.146.97')
            t1 = time.time()
            bc = BertClient(ip=BERT_IP)  # h3class_5lou
            wordvec = bc.encode(content)
            wordvec = wordvec.reshape((wordvec.shape[0], 1, wordvec.shape[1]))
            t2 = time.time()
            print("return sentence_vector by bert_as_service costs {}s".format(t2 - t1))
            t3 = time.time()
            # clear_session()
            # model = models.load_model('./keras_classify_model/keras_bert_lishi_cz.h5')
            y_pred = model.predict(wordvec)

            t4 = time.time()
            print("model predicts costs {}s".format(t4 - t3))
            chapter = 0
            for i in range(len(y_pred)):
                max_value = max(y_pred[i])
                for j in range(len(y_pred[i])):
                    if max_value == y_pred[i][j]:
                        chapter = j + 1
                        break
            know_label = know_label_[chapter]
            t5 = time.time()
            simi_tfidf = get_similarity(content[0], know_label)
            t6 = time.time()
            print("get knowledge similarity costs {}s".format(t6 - t5))
            point_list = get_point_list(know_label, simi_tfidf)
            points = sorted(point_list, key=lambda k: k[1], reverse=True)[0:50]
            # 返回题干的分词
            ques_seg = []
            stopwords = stop_words_list
            words = jieba.cut(content[0])
            for item in words:
                # if item not in stopwords and len(item) >= 2:
                if len(item) >= 2:
                    ques_seg.append(item)
            points_str = get_return_(ques_seg, points)
        else:
            json_data = get_tree_data(subject_id, headers)
            json_data = json_data.get("data")
            know_label = per_knowledge_(json_data)
            simi_tfidf = []
            # 词频相似度
            for lal in know_label:
                sim = get_similarity(content, lal)
                for simi in sim:
                    simi_tfidf.append(simi)
            point_list = get_point_list_(know_label, simi_tfidf)
            points = sort(point_list)[0:50]
            # 返回题干的分词
            ques_seg = []
            stopwords = stop_words_list
            words = jieba.cut(content)
            for item in words:
                # if item not in stopwords and len(item) >= 2:
                if len(item) >= 2:
                    ques_seg.append(item)
            points_str = get_return(ques_seg, points)
        return points_str


def train_kerasbert(ques_path, subject_id):
    """
    :param ques_path: 测试题目的数据
    :param json_path: 知识点的json数据路径
    :return:
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko',
        "Content-Type": "application/json",
        'Connection': 'close',
        'Authorization': 'Potent {} {}'.format(DEFAULT_CLIENT_ID, DEFAULT_CLIENT_SECRET)
    }
    json_data = get_tree_data(subject_id, headers)
    know_label = per_knowledge_(json_data)
    con, knows, know_label, one_dict = pre_test(ques_path, know_label=know_label)
    length, contents = get_data(one_dict)

    # token_dict = get_token_dict(dict_path)
    # [X1, X2] = get_encode(contents, token_dict)
    # wordvec = build_bert_model(X1, X2)
    # bc = BertClient(ip='118.24.146.97')
    bc = BertClient(ip=BERT_IP)  # h3class_5lou
    wordvec = bc.encode(contents)
    wordvec = wordvec.reshape((wordvec.shape[0], 1, wordvec.shape[1]))

    # 标签类
    label_count = len(length)  # 类别数目
    data = []
    if length:
        data.append(np.zeros(length[0], dtype=int))
    for i in range(1, len(length)):
        data.append(np.ones(length[i], dtype=int) * i)

    # y = np.concatenate((np.zeros(label1, dtype=int), np.ones(label2, dtype=int) * 2, np.ones(label3, dtype=int) * 1,
    #                     np.ones(label4, dtype=int) * 4, np.ones(label5, dtype=int) * 3))
    # 					# np.ones(37, dtype=int)*6,
    # 					# np.zeros(3, dtype=int), np.ones(0, dtype=int)*7))
    y = np.concatenate(data)
    y = to_categorical(y, num_classes=label_count)
    # # p = Dense(2, activation='sigmoid')(x)
    X_train, X_test, Y_train, Y_test = train_test_split(wordvec, y, test_size=0.10, random_state=42)
    train(X_train, Y_train, label_count)


# if __name__ == '__main__':
#     con, knows, know_label, one_dict = pre_test()
#     length, contents = get_data(one_dict)
#     token_dict = get_token_dict(dict_path)
#     # # get_encode()
#     [X1, X2] = get_encode(contents, token_dict)
#     wordvec = build_bert_model(X1, X2)
#     # # 标签类
#     y = np.concatenate((np.zeros(399, dtype=int), np.ones(202, dtype=int) * 2, np.ones(214, dtype=int) * 1,
#                         np.ones(78, dtype=int) * 4, np.ones(79, dtype=int) * 3))
#     # 					# np.ones(37, dtype=int)*6,
#     # 					# np.zeros(3, dtype=int), np.ones(0, dtype=int)*7))
#     y = to_categorical(y, num_classes=5)
#     # # p = Dense(2, activation='sigmoid')(x)
#     X_train, X_test, Y_train, Y_test = train_test_split(wordvec, y, test_size=0.10, random_state=42)
#     train(X_train, Y_train)
#
# 	with open('keras_bert_lishi_cz.yml', 'r') as f:
# 		yaml_string = yaml.load(f)
# 	model = model_from_yaml(yaml_string)
# 	model.load_weights('keras_bert_lishi_cz.h5')
#
# 	y_pred = model.predict(X_test)
# 	for i in range(len(y_pred)):
# 		max_value = max(y_pred[i])
# 		chapter = 0
# 		for j in range(len(y_pred[i])):
# 			if max_value == y_pred[i][j]:
# 				y_pred[i][j] = 1
# 			else:
# 				y_pred[i][j] = 0
# 	print('accuracy %s' % accuracy_score(y_pred, Y_test))

# [X1, X2] = get_encode(string, token_dict)
# wordvec = build_bert_model(X1, X2)
# y_string = model.predict(wordvec)
# string = [con[0], con[0]]
# points_str = question_classify("", "", know_label, item=None, string=string)
