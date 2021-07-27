#!/bin/bash

# ANN_6K_1K
echo "Training model Ann_6k_1k, lr = 1e-3, epochs = 30, batch_size = 256"
python train.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 20 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_1k \
        --model_name ANN_6k_1k_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_4k_4k, lr = 1e-4, epochs = 30, batch_size = 256"
python train.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 20 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_4k \
        --model_name ANN_4k_4k_1e4_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_5k_1k, lr = 1e-5, epochs = 30, batch_size = 256"
python train.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 20 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_1k \
        --model_name ANN_5k_1k_1e5_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_3K_3K
echo "Training model Ann_3k_3k, lr = 1e-3, epochs = 30, batch_size = 256"
python train.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 20 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_3k \
        --model_name ANN_3k_3k_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" 

P4=$!
wait $P1 $P2 $P3 $P4
