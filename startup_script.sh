#!/bin/bash

export CODE_DIR=/workspace

gcsfuse -o allow_other -file-mode=755 -dir-mode=755 roberta_processed_corpus /data
gcsfuse -o allow_other -file-mode=777 -dir-mode=777 qywu-pretrain-roberta-bucket $CODE_DIR/LargePretrain/outputs

cd $CODE_DIR/LargePretrain
git checkout gcp_exp1
git pull

docker run --gpus all --ipc=host -itd \
                      --network=host \
                      --mount type=bind,src=/data,dst=/data \
                      --mount type=bind,src=$CODE_DIR/LargePretrain,dst=/workspace/LargePretrain \
                      pretrain

# docker run --gpus all --ipc=host -it \
#                       --network=host \
#                       --mount type=bind,src=/data,dst=/data \
#                       --mount type=bind,src=$CODE_DIR/LargePretrain,dst=/workspace/LargePretrain \
#                       pretrain

# cd outputs
# touch there_is_error.txt
# reboot
