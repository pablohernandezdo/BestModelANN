#!/bin/bash

echo "Training model Ann_6k_3k, lr = 1e-5, epochs = 30, batch_size = 256"
python train.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_3k \
        --model_name ANN_6k_3k_1e5_256_30epch \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"
