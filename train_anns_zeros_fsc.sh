#!/bin/bash

# ANN_6K_6K
echo "Training model Ann_6k_6k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_6k \
        --model_name ANN_6k_6k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_6k_6k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_6k \
        --model_name ANN_6k_6k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_6k_6k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_6k \
        --model_name ANN_6k_6k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_6K_5K
echo "Training model Ann_6k_5k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_5k \
        --model_name ANN_6k_5k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P4=$!

echo "Training model Ann_6k_5k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_5k \
        --model_name ANN_6k_5k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P5=$!

echo "Training model Ann_6k_5k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_5k \
        --model_name ANN_6k_5k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6

# ANN_6K_4K
echo "Training model Ann_6k_4k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_4k \
        --model_name ANN_6k_4k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_6k_4k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_4k \
        --model_name ANN_6k_4k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_6k_4k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_4k \
        --model_name ANN_6k_4k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_6K_3K
echo "Training model Ann_6k_3k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_3k \
        --model_name ANN_6k_3k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P4=$!

echo "Training model Ann_6k_3k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_3k \
        --model_name ANN_6k_3k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P5=$!

echo "Training model Ann_6k_3k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_3k \
        --model_name ANN_6k_3k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6

# ANN_6K_2K
echo "Training model Ann_6k_2k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_2k \
        --model_name ANN_6k_2k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_6k_2k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_2k \
        --model_name ANN_6k_2k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_6k_2k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_2k \
        --model_name ANN_6k_2k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_6K_1K
echo "Training model Ann_6k_1k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_1k \
        --model_name ANN_6k_1k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P4=$!

echo "Training model Ann_6k_1k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_1k \
        --model_name ANN_6k_1k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P5=$!

echo "Training model Ann_6k_1k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_6k_1k \
        --model_name ANN_6k_1k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6

# ANN_5K_5K
echo "Training model Ann_5k_5k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_5k \
        --model_name ANN_5k_5k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_5k_5k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_5k \
        --model_name ANN_5k_5k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_5k_5k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_5k \
        --model_name ANN_5k_5k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_5K_4K
echo "Training model Ann_5k_4k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_4k \
        --model_name ANN_5k_4k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P4=$!

echo "Training model Ann_5k_4k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_4k \
        --model_name ANN_5k_4k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P5=$!

echo "Training model Ann_5k_4k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_4k \
        --model_name ANN_5k_4k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6

# ANN_5K_3K
echo "Training model Ann_5k_3k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_3k \
        --model_name ANN_5k_3k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_5k_3k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_3k \
        --model_name ANN_5k_3k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_5k_3k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_3k \
        --model_name ANN_5k_3k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_5K_2K
echo "Training model Ann_5k_2k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_2k \
        --model_name ANN_5k_2k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P4=$!

echo "Training model Ann_5k_2k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_2k \
        --model_name ANN_5k_2k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P5=$!

echo "Training model Ann_5k_2k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_2k \
        --model_name ANN_5k_2k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6

# ANN_5K_1K
echo "Training model Ann_5k_1k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_1k \
        --model_name ANN_5k_1k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_5k_1k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_1k \
        --model_name ANN_5k_1k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_5k_1k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_5k_1k \
        --model_name ANN_5k_1k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_4K_4K
echo "Training model Ann_4k_4k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_4k \
        --model_name ANN_4k_4k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P4=$!

echo "Training model Ann_4k_4k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_4k \
        --model_name ANN_4k_4k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P5=$!

echo "Training model Ann_4k_4k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_4k \
        --model_name ANN_4k_4k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6

# ANN_4K_3K
echo "Training model Ann_4k_3k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_3k \
        --model_name ANN_4k_3k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_4k_3k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_3k \
        --model_name ANN_4k_3k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_4k_3k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_3k \
        --model_name ANN_4k_3k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_4K_2K
echo "Training model Ann_4k_2k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_2k \
        --model_name ANN_4k_2k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P4=$!

echo "Training model Ann_4k_2k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_2k \
        --model_name ANN_4k_2k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P5=$!

echo "Training model Ann_4k_2k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_2k \
        --model_name ANN_4k_2k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6

# ANN_4K_1K
echo "Training model Ann_4k_1k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_1k \
        --model_name ANN_4k_1k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_4k_1k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_1k \
        --model_name ANN_4k_1k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_4k_1k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_4k_1k \
        --model_name ANN_4k_1k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_3K_3K
echo "Training model Ann_3k_3k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_3k \
        --model_name ANN_3k_3k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P4=$!

echo "Training model Ann_3k_3k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_3k \
        --model_name ANN_3k_3k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P5=$!

echo "Training model Ann_3k_3k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_3k \
        --model_name ANN_3k_3k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6

# ANN_3K_2K
echo "Training model Ann_3k_2k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_2k \
        --model_name ANN_3k_2k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_3k_2k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_2k \
        --model_name ANN_3k_2k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_3k_2k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_2k \
        --model_name ANN_3k_2k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_3K_1K
echo "Training model Ann_3k_1k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_1k \
        --model_name ANN_3k_1k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P4=$!

echo "Training model Ann_3k_1k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_1k \
        --model_name ANN_3k_1k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P5=$!

echo "Training model Ann_3k_1k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_1k \
        --model_name ANN_3k_1k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6

# ANN_2K_2K
echo "Training model Ann_2k_2k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_2k_2k \
        --model_name ANN_2k_2k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_2k_2k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_2k_2k \
        --model_name ANN_2k_2k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_2k_2k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_2k_2k \
        --model_name ANN_2k_2k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P3=$!

# ANN_2K_1K
echo "Training model Ann_2k_1k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_2k_1k \
        --model_name ANN_2k_1k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P4=$!

echo "Training model Ann_2k_1k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_2k_1k \
        --model_name ANN_2k_1k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P5=$!

echo "Training model Ann_2k_1k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_2k_1k \
        --model_name ANN_2k_1k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P6=$!
wait $P1 $P2 $P3 $P4 $P5 $P6

# ANN_1K_1K
echo "Training model Ann_1k_1k, lr = 1e-3, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-3 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_1k_1k \
        --model_name ANN_1k_1k_1e3_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P1=$!

echo "Training model Ann_1k_1k, lr = 1e-4, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-4 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_1k_1k \
        --model_name ANN_1k_1k_1e4_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy" &

P2=$!

echo "Training model Ann_1k_1k, lr = 1e-5, epochs = 30, batch_size = 256"
python train_fsc.py \
        --lr 1e-5 \
        --device 2 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_1k_1k \
        --model_name ANN_1k_1k_1e5_256_fsc \
        --dataset_name "STEAD-ZEROS" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"

P3=$!
wait $P1 $P2 $P3