FROM nvcr.io/nvidia/pytorch:20.02-py3

# TorchFly
RUN git clone https://github.com/qywu/TorchFly.git && \
    cd TorchFly && \
    pip install -r requirements.txt

# Pretraining Code
# RUN git clone  https://github.com/qywu/LargePretrain.git
COPY ./entrypoint.sh .
# COPY . LargePretrain

# RUN conda init bash
# RUN conda activate torchfly

# Environment Variables
ENV CODE_DIR=/workspace/LargePretrain
ENV DATA_DIR=/data

ENV PATH /opt/conda/envs/env/bin:$PATH

# Install TorchFly
RUN cd TorchFly && \
    git pull && \
    pip install -e .

SHELL ["/bin/bash", "-c"]

ENTRYPOINT [ "/workspace/entrypoint.sh"]
