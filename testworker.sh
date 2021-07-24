#!/bin/bash

# ANN_6K_6K
echo "Training model Ann_6k_6k, lr = 1e-3, epochs = 20, batch_size = 256"
python train.py \
        --lr 1e-3 \
        --device 2 \
        --workers 4 \
        --epochs 20 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier ANN_6k_6k \
        --model_name ANN_6k_6k_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"