#!/usr/bin/env bash

# main function
python train.py --log_name PCB_PerLoss  --gpu_nums 1 --batch_size 16  --worker_num 8 --reid_model "PCB_PerLoss"