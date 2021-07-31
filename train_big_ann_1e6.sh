#!/bin/bash

# ANN_6K_6K
echo "Training model Ann_6k_6k, lr = 1e-6, epochs = 50, batch_size = 256"
python train_fsc.py \
        --lr 1e-6 \
        --device 2 \
        --epochs 50 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_6k \
        --model_name ANN_6k_6k_1e3_256_fsc_50epch \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" 