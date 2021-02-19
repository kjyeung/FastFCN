#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.train --dataset ade20k \
            --model encnet --jpu --aux --se-loss --no-val \
            --backbone resnet50 --checkname encnet_res50_ade20k_train

#test [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset ade20k \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume /home/kaijie/workspace/FastFCN/experiments/segmentation/runs/ade20k/encnet/encnet_res50_ade20k_train/checkpoint_119.pth.tar \
     --split val --mode testval

#test [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset ade20k \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode testval --ms

#predict [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset ade20k \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode test

#predict [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset ade20k \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode test --ms