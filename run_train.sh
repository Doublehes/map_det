#!/bin/bash
cd /home/flow/Code/MilitaryModel/maptr
export PYTHONUNBUFFERED=1
/home/flow/Application/Anaconda3/envs/MapTracker/bin/python -u train.py --work-dir ./work_dirs/maptr > work_dirs/maptr/train.log 2>&1
