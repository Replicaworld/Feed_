#!/usr/bin/env bash
FEED_DIR="myfeed"
cd $FEED_DIR

if [ ! -d "Dump" ] ; then
    echo "Creating Dump dir: 'Dump/'"
    mkdir Dump
fi

if [ ! -d "logs" ] ; then
    echo "Creating logs dir: 'logs/'"
    mkdir logs
fi

python3 utils.py --job_name update_device_hash_dict
python3 utils.py --job_name update_item_hash_dict
spark-submit collab_semantic.py $(date "+%Y-%m-%d") &> out
log_fname = $(date "+%Y-%m-%d").log
gsutil cp -r /home/Avatar/$FEED_DIR/Dump/collab_mse_model/* gs://nis-dataproc/tf_models/en/collab_mse_model/
gsutil cp /home/Avatar/$FEED_DIR/logs/$log_fname gs://nis-dataproc/data/inshorts_myfeed_training/collab_mse_logs/$log_fname
gsutil cp /home/Avatar/$FEED_DIR/out gs://nis-dataproc/data/inshorts_myfeed_training/collab_mse_logs/$(date "+%Y-%m-%d").out
