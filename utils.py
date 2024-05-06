import argparse
import datetime
import logging
import numpy as np
import os
import re
import sys
from tqdm import tqdm
from pymongo import MongoClient
from redis import StrictRedis
import ujson


REDIS_HOST = '172.16.11.61'
REDIS_PORT = 6379
REDIS_DEVICE_DB = 1
DATE_FMT = "%Y-%m-%d"
BASE_PATH = os.getcwd()
DEVICE_HASH_DICT_PATH = BASE_PATH + '/Dump/deviceHashMap.json'
ITEM_HASH_DICT_PATH = BASE_PATH + '/Dump/itemHashMap.json'

MODULE_JOBS = ['update_device_hash_dict', 'update_item_hash_dict']

logger = logging.getLogger("Myfeed_Jobs")
hdlr = logging.FileHandler(
    BASE_PATH + '/logs/myfeed_jobs.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.DEBUG)


def load_dict(dict_path):
    dict_ = {}
    try:
        logger.info("Loading dict" + dict_path)
        with open(dict_path, 'r') as handle:
            dict_ = ujson.load(handle)

        logger.info("dict successfully loaded")
    except Exception as e:
        logger.info("No old dict exist.")
    return dict_


def save_dict(dict_, dict_path):
    logger.info("Dumping dict" + dict_path)
    with open(dict_path, 'w') as handle:
        ujson.dump(dict_, handle)

    logger.info("dict successfully dumped")


def normalize_text(text):
    return re.sub(r"([\w+$\s-]+|[^\w+$\s-]+)\s*", r"\1 ", text.lower().rstrip('.\n'))


def chunks(l, n):
    n = max(1, n)
    for i in range(0, len(l), n):
        yield l[i: i + n]


def categorise(l, n_classes=2):
    vals = dict(enumerate(set(l)))
    out = []
    for i in l:
        a = np.zeros(n_classes)
        a[vals[i]] = 1
        out.append(a)
    return out


def get_mongo_coll(db_name='nis-news', coll_name='News'):
    hosts = ['172.16.11.196', '172.16.11.195', '172.16.11.194']
    host = ','.join([i + ":27017" for i in hosts])
    conn_url = 'mongodb://root:superman@' + host + '/' + db_name
    client = MongoClient(conn_url)
    return client[db_name][coll_name]


def update_device_hash_dict():
    red = StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DEVICE_DB)

    device_hash_dict = load_dict(DEVICE_HASH_DICT_PATH)
    logger.info("Current devices in dict: " + str(len(device_hash_dict)))
    last_update_time = device_hash_dict.get("updateTime",
                                            (datetime.datetime.now() - datetime.timedelta(days=3)).strftime(DATE_FMT))
    last_update_time = datetime.datetime.strptime(last_update_time, DATE_FMT)
    dates = [(last_update_time + datetime.timedelta(days=i)).strftime("%Y-%m-%d")
             for i in range((datetime.datetime.now() - last_update_time).days + 1)]
    new_devices = set([])
    for date in dates:
        new_devices_date = red.smembers("DAY_WISE_DEVICE_V3:" + date)
        if new_devices_date:
            new_devices_date = {d.decode('utf-8') for d in new_devices_date}
            new_devices = new_devices.union(new_devices_date)
    new_devices = list(new_devices)
    logger.info("News devices count: " + str(len(new_devices)))

    device_chunks = chunks(new_devices, 10000)
    for chunk in device_chunks:
        pipe = red.pipeline()
        for d in chunk:
            pipe.get('DEVICE_HASH_V3:' + d)

        device_hashes = pipe.execute()

        for i in range(len(chunk)):
            if device_hashes[i]:
                device_hash_dict[chunk[i]] = int(device_hashes[i])

        logger.info("After updation, devices in dict: " +
                    str(len(device_hash_dict)))
    device_hash_dict["updateTime"] = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(DATE_FMT)
    if len(new_devices) > 1:
        save_dict(device_hash_dict, DEVICE_HASH_DICT_PATH)


def update_item_hash_dict():
    item_hash_dict = load_dict(ITEM_HASH_DICT_PATH)
    logger.info("Current items in dict: " + str(len(item_hash_dict)))
    last_update_time = item_hash_dict.get("updateTime",
                                          (datetime.datetime.now() - datetime.timedelta(days=20)).strftime(DATE_FMT))
    last_update_time = datetime.datetime.strptime(last_update_time, DATE_FMT)
    item_coll = get_mongo_coll('nis-news', 'NewsGroup')
    cur = item_coll.find({"createdAt": {"$gt": last_update_time}, "tenant": "ENGLISH"})
    new_items = set([])
    for c in tqdm(cur):
        if 'newsHash' in c:
            item_hash_dict[c["_id"]] = c["newsHash"]
            new_items.add(c["_id"])

    logger.info("News items count: " + str(len(new_items)))
    logger.info("After updation, items in dict: " +
                str(len(item_hash_dict)))
    item_hash_dict["updateTime"] = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime(DATE_FMT)
    if len(new_items) > 1:
        save_dict(item_hash_dict, ITEM_HASH_DICT_PATH)


if __name__ == "__main__":

    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Additional Jobs for myfeed')

    parser.add_argument("--job_name", type=str,
                        help="Choose from: " + str(MODULE_JOBS))

    args = parser.parse_args()

    job_name = args.job_name
    if job_name not in MODULE_JOBS:
        logger.error("Job not chosen from job list, exiting..")
        sys.exit("Choose from: " + str(MODULE_JOBS))

    if job_name == "update_device_hash_dict":
        logger.info("Updating device_hash_dict")
        update_device_hash_dict()
        logger.info("device_hash_dict successfully updated")

    if job_name == "update_item_hash_dict":
        logger.info("Updating item_hash_dict")
        update_item_hash_dict()
        logger.info("item_hash_dict successfully updated")