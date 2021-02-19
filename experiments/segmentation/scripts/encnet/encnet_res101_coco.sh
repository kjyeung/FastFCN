#!/usr/bin/env bash

#train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.train --dataset coco \
            --model encnet --jpu --aux --se-loss --no-val \
            --backbone resnet101 --checkname encnet_res101_coco

#test [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset coco \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume ${MODEL} --split val --mode testval

#test [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset coco \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode testval --ms

#predict [single-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset coco \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode test

#predict [multi-scale]
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m experiments.segmentation.test --dataset coco \
    --model encnet --jpu --aux --se-loss \
    --backbone resnet101 --resume {MODEL} --split val --mode test --ms