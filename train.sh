export WANDB_PROJECT=biokgbind

# train with single-gpu
export CUDA_VISIBLE_DEVICES=1
python train_bind.py \
    --batch_size 4096 \
    --n_epoch 10 \
    --n_layer 6 \
    --save_dir "./checkpoints/bind" \
    --split_dir "./data/BindData/train_test_split" 
    # > log_bind.txt 2>&1 &

# train with multi-gpu
# torchrun --nproc_per_node 8 train_bind.py \
#     --batch_size 4096 \
#     --n_epoch 50 \
#     --n_layer 6 \
#     --save_dir "./checkpoints/bind-50epoch" \
#     --split_dir "./data/BindData/train_test_split" \
#     --dataloader_num_workers 4 \
#     --use_wandb True \
#     --eval_steps 200 \
#     --learning_rate 5e-4 \
#     > log_bind.txt 2>&1 &
