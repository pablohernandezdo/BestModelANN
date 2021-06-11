#!/bin/bash

echo "Training model Ann_6k_6k, lr = 1e-3, epochs = 5, batch_size = 256"
python train.py \
        --lr 1e-3 \
        --epochs 5 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier ANN_6k_6k \
        --model_name ANN_6k_6k_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/Train_constant.npy" \
        --val_path "Data/TrainReady/Val_constant.npy" &

P1=$!

echo "Training model Ann_6k_6k, lr = 1e-4, epochs = 5, batch_size = 256"
python train.py \
        --lr 1e-4 \
        --epochs 5 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier ANN_6k_6k \
        --model_name ANN_6k_6k_1e4_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/Train_constant.npy" \
        --val_path "Data/TrainReady/Val_constant.npy" &

P2=$!

echo "Training model Ann_6k_6k, lr = 1e-5, epochs = 5, batch_size = 256"
python train.py \
        --lr 1e-5 \
        --epochs 5 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier ANN_6k_6k \
        --model_name ANN_6k_6k_1e5_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/Train_constant.npy" \
        --val_path "Data/TrainReady/Val_constant.npy" &

P3=$!

echo "Training model Ann_6k_5k, lr = 1e-3, epochs = 5, batch_size = 256"
python train.py \
        --lr 1e-3 \
        --epochs 5 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier ANN_6k_5k \
        --model_name ANN_6k_5k_1e3_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/Train_constant.npy" \
        --val_path "Data/TrainReady/Val_constant.npy" &

P4=$!

echo "Training model Ann_6k_5k, lr = 1e-4, epochs = 5, batch_size = 256"
python train.py \
        --lr 1e-4 \
        --epochs 5 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier ANN_6k_5k \
        --model_name ANN_6k_5k_1e4_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/Train_constant.npy" \
        --val_path "Data/TrainReady/Val_constant.npy" &

P5=$!

echo "Training model Ann_6k_5k, lr = 1e-5, epochs = 5, batch_size = 256"
python train.py \
        --lr 1e-5 \
        --epochs 5 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 1 \
        --model_folder 'models'  \
        --classifier ANN_6k_5k \
        --model_name ANN_6k_5k_1e5_256 \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/Train_constant.npy" \
        --val_path "Data/TrainReady/Val_constant.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6
