#!/usr/bin/env bash
instance_name="nis-feed-training-ind"

echo "Starting instance $instance_name"
gcloud compute instances start $instance_name --zone=us-east1-c

instanceip=$(sudo gcloud compute instances list | grep $instance_name | egrep -o "172.16.[0-9]+.[0-9]+")
echo $instanceip
ssh-keygen -f "/home/Avatar/.ssh/known_hosts" -R $instanceip
ssh -o "StrictHostKeyChecking no" $instanceip "bash -s" < /home/Avatar/inshorts-myfeed-training/training.sh

echo "Stopping instance $instance_name..."
gcloud compute instances stop $instance_name --zone=us-east1-c
echo "Instance stopped"