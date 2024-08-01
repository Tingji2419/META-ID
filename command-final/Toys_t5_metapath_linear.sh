#!/bin/bash

DATASET=Toys
ITEM_INDEXING=metapath
BACKBONE=t5-small

for seed in 0 1 2 3 4 
do

cur_timestr=$(date "+%Y%m%d%H%M%S")
dir_path="../log-final/$DATASET/"

if [ ! -d "$dir_path" ]; then
    mkdir -p "$dir_path"
fi
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=5234 ../src/train.py \
    --item_indexing $ITEM_INDEXING \
    --tasks sequential,straightforward \
    --datasets $DATASET \
    --epochs 10 \
    --batch_size 128 \
    --backbone $BACKBONE \
    --cutoff 1024 \
    --random_initialize 0 \
    --linear \
    --seed $seed \
    --linear_alpha 0.1 \
    --metapath_cluster_num 200 \
    --log_dir ../log-final \
    --time_str $cur_timestr \
    > ../log-final/$DATASET/$cur_timestr\_$DATASET\_$BACKBONE\_$ITEM_INDEXING.log
done
# > ../log-linear/$DATASET/$DATASET\_$BACKBONE\_$ITEM_INDEXING.log
