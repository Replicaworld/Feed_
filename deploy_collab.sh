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


python3 /home/Avatar/$FEED_DIR/utils.py --job_name update_device_hash_dict
python3 /home/Avatar/$FEED_DIR/utils.py --job_name update_item_hash_dict
#spark-submit /home/Avatar/$FEED_DIR/collab_semantic.py $(date "+%Y-%m-%d") 15 &> /home/Avatar/$FEED_DIR/out
spark-submit /home/Avatar/$FEED_DIR/collab_semantic.py &> /home/Avatar/$FEED_DIR/out
log_fname=$(date "+%Y-%m-%d").log
rm /home/Avatar/.gsutil/tracker-files/*
gsutil -m rsync -r /home/Avatar/$FEED_DIR/Dump/collab_mse_model gs://nis-dataproc/tf_inshorts_models/en/collab_mse_model
gsutil cp /home/Avatar/$FEED_DIR/Dump/*HashMap.json gs://nis-dataproc/tf_inshorts_models/en/data/
gsutil cp /home/Avatar/$FEED_DIR/logs/$log_fname gs://nis-dataproc/data/inshorts_feed_training/collab_mse_logs/
gsutil cp /home/Avatar/$FEED_DIR/out gs://nis-dataproc/data/inshorts_feed_training/collab_mse_logs/
