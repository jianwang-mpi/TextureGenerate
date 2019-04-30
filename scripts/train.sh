#!/usr/bin/env bash

# main function

rlaunch --cpu=9 --gpu=1 --memory=12000 -- python3 ../train.py --log_name PCB_PerLoss  --gpu_nums 1 --batch_size 16  --worker_num 8 --reid_triplet_loss_weight 0 --uvmap_intern_loss_weight 0 --reid_softmax_loss_weight 0 --reid_triplet_hard_loss_weight 0 --reid_triplet_loss_not_feature_weight 0 --perceptual_loss_weight 5000 --layer 5 --fake_and_true_loss_weight 0 --reid_model "PCB_PerLoss" --triplet 1 --epoch 120