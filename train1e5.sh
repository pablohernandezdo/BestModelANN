# ANN_6K_6K
echo "Training model Ann_3k_1k, lr = 1e-5, epochs = 30, batch_size = 256"
python train.py \
        --lr 1e-5 \
        --device 3 \
        --epochs 30 \
        --batch_size 256 \
        --earlystop 0 \
        --eval_iter 30 \
        --model_folder 'models'  \
        --classifier ANN_3k_1k \
        --model_name ANN_3k_1k_30epch \
        --dataset_name "STEAD" \
        --train_path "Data/TrainReady/train_zeros.npy" \
        --val_path "Data/TrainReady/val_zeros.npy"