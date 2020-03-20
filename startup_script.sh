#!/bin/bash
docker run --gpus all --ipc=host -it  \
                      --network=host \
                      --mount type=bind,src=/home/wuqy1203/Desktop/Projects/TorchFly/examples/LargeScalePretrain/data,dst=/data \
                      --mount type=bind,src=/home/wuqy1203/Desktop/Projects/LargePretrain,dst=/workspace/LargePretrain \
                      pretrain