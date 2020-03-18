FROM nvcr.io/nvidia/pytorch:20.02-py3

# TorchFly
RUN git clone https://github.com/qywu/TorchFly.git && \
    cd TorchFly && \
    conda env create -f environment.yml

# Pretraining Code
RUN git clone  https://github.com/qywu/LargePretrain.git 

# Environment Variables
ENV CODE_DIR=/workspace/LargePretrain
ENV SAVE_DIR=/workspace/LargePretrain/outputs
ENV DATA_DIR=/data

ENTRYPOINT [ "/workspace/LargePretrain/start_training.sh"]