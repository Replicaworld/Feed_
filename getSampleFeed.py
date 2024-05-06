import datetime
import json
import logging
import os
import pickle
import sent2vec
import shutil
import sys
import random

from numpy.core._multiarray_umath import ndarray

import utils

from google.cloud import storage
from keras.layers import Input, Embedding, Concatenate
from keras.layers.core import Flatten, Dense, Dropout
from keras.models import Model, model_from_json
from keras.optimizers import TFOptimizer
from keras.regularizers import l2
import keras.backend as K
import numpy as np
import pandas as pd
from pymongo import MongoClient
import tensorflow as tf
from tqdm import tqdm

def get_mongo_coll(db_name='nis-news', coll_name='News'):
    hosts = ['172.16.11.196', '172.16.11.195', '172.16.11.194']
    host = ','.join([i + ":27017" for i in hosts])
    conn_url = 'mongodb://root:superman@' + host + '/' + db_name
    client = MongoClient(conn_url)
    return client[db_name][coll_name]

DUMP_DIR = "Dump/"
SEN2VEC_MODEL_PATH = DUMP_DIR + "/sent2vec_model_50.bin"

# sentence vector model loading
sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model(SEN2VEC_MODEL_PATH)

device_hash_dict = utils.load_dict_pickle(utils.DEVICE_HASH_DICT_PATH)

news_group_coll = get_mongo_coll(db_name="nis-news", coll_name="NewsGroup")
news_coll = get_mongo_coll(db_name='nis-news', coll_name='News')

st = datetime.datetime(2020, 4, 25)
et = datetime.datetime(2020, 4, 26)

hash_ids = [t['_id'] + '-1' for t in news_group_coll.find({'tenant': 'ENGLISH',
                                                           'createdAt': {"$gt": st, '$lt': et}},
                                                          {'_id': 1})]
cursor = news_coll.find({'_id': {'$in': hash_ids}}, {'content': 1})
item_vector_map = {j['_id']: sent2vec_model.embed_sentence(utils.normalize_text(j['content'])) for j in cursor}

cursor = news_coll.find({'_id': {'$in': hash_ids}}, {'title': 1, 'categories':1})
item_title_map = {j['_id']: (j['title'], "coronavirus" in "##".join(j["categories"]).lower()) for j in cursor}


# MODEL_JSON_FILENAME = "model.json"

# MODEL_WEIGHTS_FILENAME = "model.h5"

# def load_model(model_dir):
#     if os.path.exists(model_dir):
#         with open(model_dir + MODEL_JSON_FILENAME, "rb") as model_json:
#             model = model_from_json(model_json.read())

#         model.load_weights(model_dir + MODEL_WEIGHTS_FILENAME)

#         return model
#     return None

def load_model(model_dir, sess):

    if os.path.exists(model_dir):
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], model_dir)

        # input_dict = {
        #     "user_inputs": sess.graph.get_tensor_by_name("user_inputs:0"),
        #     "item_inputs": sess.graph.get_tensor_by_name("item_semantic_vector:0")
        # }
        #
        # output_dict = {
        #     "prediction:0": sess.graph.get_tensor_by_name("classifier/class_out/Softmax:0")
        # }

    return sess #, input_dict, output_dict


sess = tf.Session(graph=tf.Graph())
sess = load_model("/root/myfeed/Dump/cs_model/", sess)
prediction = sess.graph.get_tensor_by_name("regressor/timespent_out/Relu:0")

DEVICEIDS = ["4621e9f6-9156-4b6c-940b-c02f4a339126", "40ab1b9b-5b78-49a2-bf04-1eb673385690",
             "62039254-6816-42d6-8690-4e0a7ea36737"]
# devices = [int(device_hash_dict[d]) for d in DEVICEIDS]
nn = set([])
ss = []
for device in DEVICEIDS:
    print device
    items_ids = item_vector_map.keys()
    device = int(device_hash_dict[device])

    u = np.array([device] * len(item_title_map))
    i = np.array([item_vector_map[h] for h in items_ids])  # type: ndarray

    pred = sess.run(prediction, feed_dict={
        "user_inputs:0": u,
        "item_semantic_vector:0": i})
    out = []
    for j in range(len(pred)):
        title = item_title_map[items_ids[j]]
        score = pred[j]
        out.append((title, score))

    # print device
    l = sorted(out, key=lambda x: -x[1])[5:25]
    nn = nn.union(set([t[0][0] for t in l]))
    s = len(filter(lambda x: x[0][1], l))
    ss.append(s)
    for i in l:
        print i
    print "\n\n"


sss = {}
for b in tqdm(range(21, 22)):
    news_group_coll = get_mongo_coll(db_name="nis-news", coll_name="NewsGroup")
    news_coll = get_mongo_coll(db_name='nis-news', coll_name='News')

    st = datetime.datetime(2020, 4, b)
    et = datetime.datetime(2020, 4, b+1)

    hash_ids = [t['_id'] + '-1' for t in news_group_coll.find({'tenant': 'ENGLISH',
                                                               'createdAt': {"$gt": st, '$lt': et}},
                                                              {'_id': 1})]
    cursor = news_coll.find({'_id': {'$in': hash_ids}}, {'content': 1})
    item_vector_map = {j['_id']: sent2vec_model.embed_sentence(utils.normalize_text(j['content'])) for j in cursor}

    cursor = news_coll.find({'_id': {'$in': hash_ids}}, {'title': 1, 'categories':1})
    item_title_map = {j['_id']: (j['title'], "coronavirus" in "##".join(j["categories"]).lower()) for j in cursor}
    # print "item_vector_map, item_title_map prepared"

    DEVICEIDS = random.sample(device_hash_dict.values(), 3000)
    nn = set([])
    ss = []
    for device in tqdm(DEVICEIDS):
        items_ids = item_vector_map.keys()
        device = int(device)

        u = np.array([device] * len(item_title_map))
        i = np.array([item_vector_map[h] for h in items_ids])  # type: ndarray

        pred = sess.run(prediction, feed_dict={
            "user_inputs:0": u,
            "item_semantic_vector:0": i})
        out = []
        for j in range(len(pred)):
            title = item_title_map[items_ids[j]]
            score = pred[j]
            out.append((title, score))

        # print device
        l = sorted(out, key=lambda x: -x[1])[3:13]
        nn = nn.union(set([t[0][0] for t in l]))
        s = len(filter(lambda x: x[0][1], l))
        ss.append(s)

    sts = len(filter(lambda x: x[1], item_title_map.values()))
    sss[b] = (np.mean(ss), sts, round(np.mean(ss)*100/sts))


pub = {}
for b in tqdm(range(1, 26)):
    news_group_coll = get_mongo_coll(db_name="nis-news", coll_name="NewsGroup")
    news_coll = get_mongo_coll(db_name='nis-news', coll_name='News')

    st = datetime.datetime(2020, 4, b)
    et = datetime.datetime(2020, 4, b+1)

    hash_ids = [t['_id'] + '-1' for t in news_group_coll.find({'tenant': 'ENGLISH',
                                                               'createdAt': {"$gt": st, '$lt': et}},
                                                              {'_id': 1})]
    cursor = news_coll.find({'_id': {'$in': hash_ids}, 'publishGroupList.countryCode': 'IN', 'newsState': {'$exists': False}, 'deleted':False, '$or' : [{'isAutoGen': False}, {'isAutoGen': None}, {'isAutoGen': {'$exists': False}}]}, {'content': 1})
    item_vector_map = {j['_id']: sent2vec_model.embed_sentence(utils.normalize_text(j['content'])) for j in cursor}

    cursor = news_coll.find({'_id': {'$in': hash_ids}, 'publishGroupList.countryCode': 'IN', 'newsState': {'$exists': False}, 'deleted':False, '$or' : [{'isAutoGen': False}, {'isAutoGen': None}, {'isAutoGen': {'$exists': False}}]}, {'title': 1, 'categories':1})
    item_title_map = {j['_id']: (j['title'], "coronavirus" in "##".join(j["categories"]).lower()) for j in cursor}
    all = len(item_title_map)
    cor = len(filter(lambda x: x[1], item_title_map.values()))
    pub[b] = (all, round(cor*100./all, 0))
    # print "item_vector_map, item_title_map prepared"



#
# with tf.Session(graph=tf.Graph()) as sess:
#     sess = load_model("/root/myfeed/Dump/cs_model/", sess)
#     prediction = sess.graph.get_tensor_by_name("classifier/class_out/Softmax:0")
#
#     # devices = [664126, 2910894]
#     for device in devices:
#         items_ids = item_vector_map.keys()
#
#         u = np.array([device] * len(item_title_map))
#         i = np.array([item_vector_map[h] for h in items_ids])  # type: ndarray
#
#         pred = sess.run(prediction, feed_dict={
#                                         "user_inputs:0": u,
#                                         "item_semantic_vector:0": i})
#         out = []
#         for j in range(len(pred)):
#             title = item_title_map[items_ids[j]]
#             score = pred[j][1]
#             out.append((title, score))
#
#         print device
#         print sorted(out, key=lambda x: -x[1])[:20]
#         print "\n\n"
