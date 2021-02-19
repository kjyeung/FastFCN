#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.train --dataset citys \
    --model encnet --jpu --aux --se-loss --no-val \
    --backbone resnet50 --checkname encnet_res50_citys

#test [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume ${MODEL} --split val --mode testval

#test [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode testval --ms

#predict [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode test

#predict [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50 --resume {MODEL} --split val --mode test --ms

#fps
CUDA_VISIBLE_DEVICES=1 python -m experiments.segmentation.test_fps_params --dataset citys \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet50