#!/bin/bash

echo "Training model Ann_6k_6k, lr = 1e-3, epochs = 5, batch_size = 128"
python train.py \
        --lr 1e-3 \
        --epochs 5 \
        --batch_size 128 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier Ann_6k_6k \
        --model_name ANN_6k_6k_1e3_128 \
        --dataset_name "STEAD" \
        --train_path "Data/TrainReady/STEAD-STEAD_0.8_train.npy" \
        --val_path "Data/TrainReady/STEAD-STEAD_0.1_val.npy"