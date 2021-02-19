#!/usr/bin/env bash

#train JPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.train --dataset citys \
    --model encnet --jpu JPU --aux --se-loss \
    --backbone resnet101 --checkname encnet_res101_jpu_citys_train
#train JPU_X
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.train --dataset citys \
    --model encnet --jpu [JPU|JPU_X] --aux --se-loss \
    --backbone resnet101 --checkname encnet_res101_citys_train

#test [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model encnet --jpu [JPU|JPU_X] --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode testval

#test [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model encnet --jpu [JPU|JPU_X] --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode testval --ms

#predict [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model encnet --jpu [JPU|JPU_X] --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode test

#predict [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model encnet --jpu [JPU|JPU_X] --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode test --ms
