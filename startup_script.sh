#!/bin/bash

export CODE_DIR=/home/wuqy1203

gcsfuse roberta_processed_corpus /data
gcsfuse qywu-pretrain-roberta-bucket $CODE_DIR/LargePretrain/outputs

cd $CODE_DIR/LargePretrain
git pull

docker run --gpus all --ipc=host    \
                      --network=host \
                      --mount type=bind,src=/data,dst=/data \
                      --mount type=bind,src$CODE_DIR/LargePretrain,dst=/workspace/LargePretrain \
                      pretrain

docker run --gpus all --ipc=host -it  \
                      --network=host \
                      --mount type=bind,src=/data,dst=/data \
                      --mount type=bind,src=$CODE_DIR/LargePretrain,dst=/workspace/LargePretrain \
                      pretrain bash