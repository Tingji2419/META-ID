#!/bin/bash
BACKBONE=t5-small


for dataset in Beauty # Sports Toys
do


#RID
ITEM_INDEXING=random
cur_timestr=$(date "+%Y%m%d%H%M%S")
dir_path="../log-baselines/$dataset/"

if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
fi
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1534 ../src/train.py \
    --item_indexing $ITEM_INDEXING \
    --tasks sequential,straightforward \
    --datasets $dataset \
    --epochs 10 \
    --batch_size 128 \
    --backbone $BACKBONE \
    --cutoff 1024 \
    --random_initialize 1 \
    --seed 0 \
    --log_dir ../log-baselines \
    --time_str $cur_timestr \
    > ../log-baselines/$dataset/$cur_timestr\_$dataset\_$BACKBONE\_$ITEM_INDEXING.log


# SID
ITEM_INDEXING=sequential
cur_timestr=$(date "+%Y%m%d%H%M%S")
dir_path="../log-baselines/$dataset/"

if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
fi
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1634 ../src/train.py \
    --item_indexing $ITEM_INDEXING \
    --tasks sequential,straightforward \
    --datasets $dataset \
    --epochs 10 \
    --batch_size 128 \
    --backbone $BACKBONE \
    --cutoff 1024 \
    --random_initialize 1 \
    --seed 0 \
    --log_dir ../log-baselines \
    --time_str $cur_timestr \
    > ../log-baselines/$dataset/$cur_timestr\_$dataset\_$BACKBONE\_$ITEM_INDEXING.log


#CID
ITEM_INDEXING=collaborative
cur_timestr=$(date "+%Y%m%d%H%M%S")
dir_path="../log-baselines/$dataset/"

if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
fi
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1734 ../src/train.py \
    --item_indexing $ITEM_INDEXING \
    --tasks sequential,straightforward \
    --datasets $dataset \
    --epochs 10 \
    --batch_size 128 \
    --backbone $BACKBONE \
    --cutoff 1024 \
    --random_initialize 0 \
    --seed 0 \
    --collaborative_token_size 500 \
    --collaborative_cluster 20 \
    --log_dir ../log-baselines \
    --time_str $cur_timestr \
    > ../log-baselines/$dataset/$cur_timestr\_$dataset\_$BACKBONE\_$ITEM_INDEXING.log

#IID
ITEM_INDEXING=independent
cur_timestr=$(date "+%Y%m%d%H%M%S")
dir_path="../log-baselines/$dataset/"

if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
fi
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=1834 ../src/train.py \
    --item_indexing $ITEM_INDEXING \
    --tasks sequential,straightforward \
    --datasets $dataset \
    --epochs 10 \
    --batch_size 128 \
    --backbone $BACKBONE \
    --cutoff 1024 \
    --random_initialize 0 \
    --seed 0 \
    --log_dir ../log-baselines \
    --time_str $cur_timestr \
    > ../log-baselines/$dataset/$cur_timestr\_$dataset\_$BACKBONE\_$ITEM_INDEXING.log

done