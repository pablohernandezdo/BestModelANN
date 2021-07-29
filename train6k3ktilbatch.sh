#!/bin/bash

# ANN_4K_2K
echo "Training model Ann_6k_3k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_til_batch.py \
        --lr 1e-5 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --final_batch 3901 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'modelos_batches'  \
        --classifier ANN_6k_3k \
        --model_name ANN_6k_3k_1e5_256 \
        --dataset_name "STEAD" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"