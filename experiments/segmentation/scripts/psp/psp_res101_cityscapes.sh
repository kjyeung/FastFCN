#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.train --dataset citys \
    --model psp --jpu --aux --aux-weight 0.4 --no-val \
    --backbone resnet101 --checkname psp_res101_citys

#test [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model psp --jpu --aux \
    --backbone resnet101 --resume ${MODEL} --split val --mode testval

#test [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model psp --jpu --aux \
    --backbone resnet101 --resume {MODEL} --split val --mode testval --ms

#predict [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model psp --jpu --aux \
    --backbone resnet101 --resume {MODEL} --split val --mode test

#predict [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset citys \
    --model psp --jpu --aux \
    --backbone resnet101 --resume {MODEL} --split val --mode test --ms

#fps
CUDA_VISIBLE_DEVICES=5 python -m experiments.segmentation.test_fps_params --dataset citys \
    --model psp --jpu --aux \
    --backbone resnet101
