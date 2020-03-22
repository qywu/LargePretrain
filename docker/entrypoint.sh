#!/bin/bash

CODE_DIR=$CODE_DIR
DATA_DIR=$DATA_DIR
SAVE_DIR=$SAVE_DIR

source /opt/conda/etc/profile.d/conda.sh

cd /workspace/TorchFly
git pull

cd /workspace/LargePretrain
git pull

python main.py