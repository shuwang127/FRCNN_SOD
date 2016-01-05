#!/bin/bash

./tools/train_net.py --gpu 0 --solver models/VGG_CNN_M_1024_timely/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel --imdb KakouTrain | tee ./train.log

