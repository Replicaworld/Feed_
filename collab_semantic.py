import datetime
import logging
import os
import shutil
import sys
import json

import numpy as np
from pymongo import MongoClient
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from pyspark.sql.types import *
import sent2vec
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tqdm import tqdm
import utils
import ujson

# environemt constant settings

# suppress INFO logs, show will show WARNING & ERROR logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

logger = logging.getLogger(str(datetime.datetime.today().date()))
hdlr = logging.FileHandler(
    'logs/' + str(datetime.datetime.today().date()) + '.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


###


# functions

def get_mongo_coll(db_name='nis-news', coll_name='News'):
    hosts = ['172.16.11.196', '172.16.11.195', '172.16.11.194']
    host = ','.join([i + ":27017" for i in hosts])
    conn_url = 'mongodb://root:superman@' + host + '/' + db_name
    client = MongoClient(conn_url)
    return client[db_name][coll_name]


###

# neural model definition


# parameters and paths setting
N_USERS = 20000000
N_ITEMS = 300000
USER_EMBEDDING_DIM = 25
ITEM_EMBEDDING_DIM = 25
HIDDEN_LAYER_SIZE = 10
SEMANTIC_VEC_LEN = 50
BATCH_SIZE = None
TRAIN_SPLIT = 1.
TRAINING_DAYS = 60
TS_THR = 20.0
STEP_SIZE = 10000
LEARNING_RATE_TS = 1e-3
L2_REG_COEFF = 3e-6
PERF_TEST_SIZE = 10000
NB_EPOCHS = 1
DUMP_DIR = "Dump/"
MODEL_DIR = DUMP_DIR + "collab_mse_model"
SEN2VEC_MODEL_PATH = DUMP_DIR + "/sent2vec_model_50.bin"
NIS_DATA_BASE_PATH = "gs://nis-segment-datasource-v3/processed/"
NIS_RAW_DATA_BASE_PATH = "gs://inshorts-segment-raw/data/segment-raw-v5/"
NIS_OLD_DATA_BASE_PATH = "gs://nis-localytics-datasource/processed/"
##


# user mapping to indices
logger.info("Loading deviceHashMap")
with open(DUMP_DIR + 'deviceHashMap.json', 'rb') as handle:
    device_idx = ujson.load(handle)

logger.info("deviceHashMap loaded")
logger.info("No of devices: " + str(len(device_idx)) + "\n")

logger.info("Loading itemHashMap")
with open(DUMP_DIR + 'itemHashMap.json', 'rb') as handle:
    item_idx = ujson.load(handle)

logger.info("itemHashMap loaded")
logger.info("No of items: " + str(len(item_idx)) + "\n")
###
# get news features (embeddings) from mongo

# sentence vector model loading
sent2vec_model = sent2vec.Sent2vecModel()
sent2vec_model.load_model(SEN2VEC_MODEL_PATH)

logger.info("Preparing item_vector_map")
news_group_coll = get_mongo_coll(db_name="nis-news", coll_name="NewsGroup")
news_coll = get_mongo_coll(db_name='nis-news', coll_name='News')
hash_ids = [t['_id'] + '-1' for t in news_group_coll.find({
    'tenant': 'ENGLISH',
    'createdAt': {
        "$gt": datetime.datetime.now() - datetime.timedelta(days=TRAINING_DAYS)}
    }, {
        '_id': 1
    })]
cursor = news_coll.find({'_id': {'$in': hash_ids}}, {'content': 1})
item_vector_map = {j['_id'][:-2]: sent2vec_model.embed_sentence(utils.normalize_text(j['content'])) for j in cursor}
logger.info("item_vector_map prepared")


###

###
def create_model(sess):
    # model architecture definition

    logger.info("Defining Neural Model architecture")

    # with tf.device('/device:CPU:0'):
    with tf.device('/GPU:0'):
        # Embedding Layers
        user_embedding = tf.get_variable("user_embedding", shape=(N_USERS, USER_EMBEDDING_DIM), dtype=tf.float32)
        user_embedding_bias = tf.get_variable("user_embedding_bias", shape=(N_USERS, 1), dtype=tf.float32)
        user_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name="user_inputs")
        user_embed_bias = tf.nn.embedding_lookup(
            user_embedding_bias, user_inputs, name="user_embed_bias_lookup")

        user_embed = tf.nn.embedding_lookup(
            user_embedding, user_inputs, name="user_embed_lookup")

        item_embedding = tf.get_variable("item_embedding", shape=(N_ITEMS, ITEM_EMBEDDING_DIM), dtype=tf.float32)
        item_embedding_bias = tf.get_variable("item_embedding_bias", shape=(N_ITEMS, 1), dtype=tf.float32)
        item_inputs = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name="item_inputs")
        item_vector = tf.placeholder(
            tf.float32, shape=[BATCH_SIZE, SEMANTIC_VEC_LEN], name='item_semantic_vector')
        timespent_true = tf.placeholder(tf.float32, [BATCH_SIZE], name="timespent_true")

        item_embed_bias = tf.nn.embedding_lookup(
            item_embedding_bias, item_inputs, name="item_embed_bias_lookup")
        item_embed = tf.nn.embedding_lookup(
            item_embedding, item_inputs, name="item_embed_lookup")

        dot_ = tf.reduce_sum(tf.multiply(user_embed, item_embed), axis=1, name='dot')
        dot_ = tf.reshape(dot_, (-1, 1))
        add_ = dot_ + user_embed_bias + item_embed_bias
        concat1_ = tf.concat([user_embed, item_vector], axis=1)
        hidden1_out = tf.layers.Dense(10, activation=tf.nn.relu, name="hidden1_out")(concat1_)
        hidden1_out = tf.layers.BatchNormalization()(hidden1_out)
        hidden2_out = tf.layers.Dense(1, activation=tf.nn.relu, name="hidden2_out")(hidden1_out)
        concat2_ = tf.concat([add_, hidden2_out], axis=1)

    with tf.device('/GPU:1'):

        timespent_out = tf.layers.Dense(1, activation=tf.nn.relu, name="timespent_out")(concat2_)

        # losses
        vars_ = tf.trainable_variables()
        mse_reg = tf.add_n([tf.losses.mean_squared_error(timespent_true, tf.reshape(timespent_out, shape=(-1,))),
                            tf.add_n([tf.nn.l2_loss(v) for v in vars_ if 'bias' not in v.name]) * L2_REG_COEFF],
                           name='mse')

        # train_steps
        mse_train_step = tf.train.AdamOptimizer(LEARNING_RATE_TS, name="Adam_MSE").minimize(mse_reg)

    logger.info("Model defined")

    # init model
    init = tf.global_variables_initializer()
    sess.run(init)

    input_dict = {
        "user_inputs": user_inputs,
        "item_inputs": item_inputs,
        "item_semantic_vector": item_vector
    }

    output_dict = {
        "timespent_out": timespent_out
    }

    return sess, input_dict, output_dict


# model export function

def save_model(model_dir, sess, input_dict, output_dict):
    logger.info("Deleting older model")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    logger.info("Older model deleted")

    logger.info("Dumping Trained Model")

    input_dict = {
        "user_inputs": sess.graph.get_tensor_by_name("user_inputs:0"),
        "item_inputs": sess.graph.get_tensor_by_name("item_inputs:0"),
        "item_semantic_vector": sess.graph.get_tensor_by_name("item_semantic_vector:0")
    }

    output_dict = {
        "item_embed": sess.graph.get_tensor_by_name("item_embed_lookup:0"),
        "item_embed_bias": sess.graph.get_tensor_by_name("item_embed_bias_lookup:0"),
        "timespent_out": sess.graph.get_tensor_by_name("timespent_out/Relu:0")
    }

    builder = tf.saved_model.builder.SavedModelBuilder(model_dir)

    sig_def = tf.saved_model.signature_def_utils.predict_signature_def(inputs=input_dict,
                                                                       outputs=output_dict)

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING], signature_def_map={'sig_def': sig_def}, clear_devices=True)

    builder.save()
    sess.close()
    logger.info("Model successfully Dumped")


def load_model(model_dir, sess):
    logger.info("Loading Saved Model")

    if os.path.exists(model_dir):
        tf.saved_model.loader.load(
            sess, [tf.saved_model.tag_constants.SERVING], model_dir)

        input_dict = {
            "user_inputs": sess.graph.get_tensor_by_name("user_inputs:0"),
            "item_inputs": sess.graph.get_tensor_by_name("item_inputs:0"),
            "item_semantic_vector": sess.graph.get_tensor_by_name("item_semantic_vector:0")
        }

        output_dict = {
            "timespent_out": sess.graph.get_tensor_by_name("timespent_out/Relu:0")
        }

        logger.info("Model Loaded")

    else:
        logger.info("No saved model found, creating new model..")

        sess, input_dict, output_dict = create_model(sess)

        logger.info("Model created")

    return sess, input_dict, output_dict


###


###

# fetching and processing data
conf = SparkConf().setAll([('spark.driver.memory', '12g')])
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)


def get_path(dates, prefix='timeSpentFrontEvents', padding=None):
    date_fmt = "%Y/%m/%d"
    dates_ = dates + []
    if padding:
        st_date, ed_date = sorted(dates)[0], sorted(dates)[-1]
        for i in range(1, 4):
            d = (datetime.datetime.strptime(ed_date, date_fmt) + datetime.timedelta(days=i)).strftime(date_fmt)
            if d < datetime.datetime.today().strftime(date_fmt):
                dates_.append(d)
    dates_ = list(set(dates_) - {"2019/02/17", "2019/02/18", "2019/05/28", "2019/06/03",
                                 "2019/07/02", "2019/07/03", "2019/07/04"})
    paths = []
    for date in dates_:
        base_path = NIS_DATA_BASE_PATH
        if date < "2018/06/26":
            base_path = NIS_OLD_DATA_BASE_PATH
        paths.append(base_path + date + "/" + prefix + "/*.parquet")
    return paths


def process_parquet_data(training_dates):
    try:
        paths = get_path(training_dates)
        data = sqlContext.read.parquet(*paths)
        data = data.filter(~data.appName.isin(['mini', 'crux']))
        view_data = data.filter(data.deviceId != '')
        view_data = view_data.filter((view_data.shortTime > 1) & (view_data.shortTime < 60))
        time_udf = F.udf(lambda x: max(min(20., x), 0.) / 4 if (1 < x < 60) else -1., FloatType())
        gid_udf = F.udf(lambda x: x[:-2] if x else "", StringType())
        view_data = view_data.withColumn("timeSpent", time_udf("shortTime"))
        view_data = view_data.select(view_data.deviceId, gid_udf(view_data.hashId).alias('hashId'),
                                     view_data.timeSpent)
        view_data = view_data.groupby(view_data.deviceId, view_data.hashId)\
                             .agg(F.max(view_data.timeSpent).alias('timeSpent'))
        return view_data
    except Exception as e:
        logger.warning("Error processing data: " + str(e))


def get_raw_path(date, hours=None):
    paths = []
    base_path = NIS_RAW_DATA_BASE_PATH + date
    if not hours:
        return base_path + "/*/*.gz"
    for hour in hours:
        paths.append(base_path + "/" + str(hour).zfill(2) + "/*.gz")
    return ",".join(paths)


def process_raw_data(training_date, hours=None):
    def view_data_filters(x):
        x = x['properties']
        deviceid_filter = ('deviceId' in x) and (x['deviceId'] != '')
        time_filter = ('timeSpent' in x) and (1 < int(x['timeSpent']) < 60)
        return deviceid_filter and time_filter

    try:
        paths = get_raw_path(training_date, hours=hours)
        rdd = sc.textFile(paths) \
            .map(json.loads) \
            .filter(lambda x: "batch" in x).flatMap(lambda x: x["batch"]) \
            .filter(lambda x: ("event" in x) and (x["event"].lower() == "timespent-front")) \
            .filter(view_data_filters) \
            .map(lambda x: x['properties'])
        view_data = rdd.map(lambda x: (x['deviceId'], x['hashId'][:-2], max(min(20., x['timeSpent']), 0.)/4.)) \
            .toDF(['deviceId', 'hashId', 'timeSpent'])
        view_data = view_data.groupby(view_data.deviceId, view_data.hashId) \
                             .agg(F.max(view_data.timeSpent).alias('timeSpent'))
        return view_data
    except Exception as e:
        logger.warning("Error processing data: " + str(e))


# model training
end_date = datetime.datetime.today()
if len(sys.argv) > 1:
    try:
        end_date = datetime.datetime.strptime(sys.argv[1], "%Y-%m-%d")
    except:
        print("Usage: 1st argument is a date in string format YYYY-MM-DD, e.g. 2017-11-20")
        sys.exit(0)
    if len(sys.argv) > 2:
        try:
            if int(sys.argv[2]) > 0:
                TRAINING_DAYS = int(sys.argv[2])
            else:
                raise ValueError("Training days invalid.")
        except:
            print("Usage: 2nd argument is training days in [1, inf) e.g. 60")
            sys.exit(0)


TRAINING_MODE = 'RAW'

training_dates = [datetime.datetime.now().strftime("%Y/%m/%d")]
hours = filter(lambda x: x > -1, [datetime.datetime.now().hour - j for j in range(1, 3)])

if TRAINING_MODE == 'PARQUET':
    training_dates = [(end_date - datetime.timedelta(days=i)).strftime("%Y/%m/%d") for i in range(TRAINING_DAYS, 0, -1)]

logger.info("Data Cleaning and Training started")

for date in training_dates:
    try:
        logger.info("Training date: " + date)
        if TRAINING_MODE == 'PARQUET':
            view_data = process_parquet_data([date])
            view_data.cache()
            data_chunks = view_data.randomSplit([1.] * 25)
        else:
            view_data = process_raw_data(date, hours)
            view_data.cache()
            data_chunks = view_data.randomSplit([1.] * 10)
        #
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess, input_dict, output_dict = load_model(
                MODEL_DIR, sess)

            train_step_mse = sess.graph.get_operation_by_name("Adam_MSE")
            mse = sess.graph.get_tensor_by_name("mse:0")
            timespent_true = sess.graph.get_tensor_by_name("timespent_true:0")
            timespent_out = sess.graph.get_tensor_by_name("timespent_out/Relu:0")

            for i, data_df in enumerate(data_chunks):
                data_df = data_df.toPandas()
                data_df.columns = ['deviceId', 'hashId', 'timeSpent']
                logger.info("Chunk count: " + str(i + 1))
                logger.info("Data Size: " + str(len(data_df)))
                logger.info("Ratio of classes 0/1 : " +
                            str(data_df[data_df.timeSpent < 2].count() * 1. / data_df[data_df.timeSpent >= 2].count()))

                # data processing and training

                logger.info("Data processing")
                data_df = data_df[data_df.hashId.isin(set(item_vector_map.keys()))]
                data_df.deviceId = data_df.deviceId.apply(
                    lambda x: int(device_idx[x]) if x in device_idx else -1
                )
                data_df["hashIdold"] = data_df["hashId"].apply(lambda x: x)
                data_df['hashId'] = data_df['hashId'].apply(
                    lambda x: int(item_idx[x]) if x in item_idx else -1
                )
                data_df = data_df[(data_df.deviceId != -1) & (data_df.hashId != -1)]
                item_vectors = np.array([item_vector_map.get(i, [0.]*50) for i in data_df.hashIdold.tolist()],
                                        dtype=np.float32)
                logger.info("Sample data : \n" + str(data_df[:5]))

                logger.info("Data processed\n Starting Training")
                step_count = int(data_df.shape[0] / STEP_SIZE) + 1

                data_df, test_df, item_vectors, test_item_vectors = train_test_split(data_df, item_vectors,
                                                                                     test_size=0.1, random_state=42)
                for _ in range(NB_EPOCHS):
                    for j in tqdm(range(step_count)):
                        df_batch = data_df[j * STEP_SIZE: j * STEP_SIZE + STEP_SIZE]
                        batch_users, batch_items, batch_timespent = df_batch.deviceId.tolist(), \
                                                                    df_batch.hashId.tolist(), \
                                                                    df_batch.timeSpent.tolist()
                        sess.run(train_step_mse,
                                 feed_dict={
                                     "user_inputs:0": batch_users,
                                     "item_inputs:0": batch_items,
                                     "item_semantic_vector:0": item_vectors[j * STEP_SIZE: j * STEP_SIZE + STEP_SIZE],
                                     "timespent_true:0": batch_timespent
                                 })
                if len(item_vectors) > 0:
                    logger.info(date + " - Train MSE: " + str(sess.run(mse,
                                                                 feed_dict={
                                                                     "user_inputs:0": data_df.deviceId[:PERF_TEST_SIZE],
                                                                     "item_inputs:0": data_df.hashId[:PERF_TEST_SIZE],
                                                                     "item_semantic_vector:0":
                                                                         item_vectors[:PERF_TEST_SIZE],
                                                                     "timespent_true:0":
                                                                         data_df.timeSpent[:PERF_TEST_SIZE]
                                                                 })))
                    logger.info(date + " Test MSE: " + str(sess.run(mse,
                                                                    feed_dict={
                                                                        "user_inputs:0": test_df.deviceId,
                                                                        "item_inputs:0": test_df.hashId,
                                                                        "item_semantic_vector:0":
                                                                            test_item_vectors,
                                                                        "timespent_true:0":
                                                                            test_df.timeSpent
                                                                    })))
                    logger.info(date + " - Sample Time Pred: " + str(sess.run(timespent_out,
                                                                              feed_dict={
                                                                                  "user_inputs:0":
                                                                                      data_df.deviceId[:PERF_TEST_SIZE],
                                                                                  "item_inputs:0":
                                                                                      data_df.hashId[:PERF_TEST_SIZE],
                                                                                  "item_semantic_vector:0":
                                                                                      item_vectors[:PERF_TEST_SIZE]
                                                                              })[:5]) + "\n\n" + str(
                                                                                            data_df.timeSpent[:5]))
                else:
                    logger.info("No MSE: Empty Dataset")

            save_model(MODEL_DIR, sess, input_dict, output_dict)
    except Exception as e:
        logger.warning("Failed for date: %s with error: %s" % (date, str(e)))
