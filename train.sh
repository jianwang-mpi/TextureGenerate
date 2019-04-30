#!/usr/bin/env bash

# main function

rlaunch --cpu=9 --gpu=1 --memory=12000 -- python3 train.py --log_name PCB_PerLoss  --gpu_nums 1 --batch_size 16  --worker_num 8 --reid_model "PCB_PerLoss"