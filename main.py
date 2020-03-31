import os
import ray
import logging
import hydra
from hydra.utils import get_original_cwd
import numpy as np
import torch
from torchvision import datasets, transforms
from torchfly.training.trainer import Trainer
from torchfly.training.data import DataDispatcher

from dataset import FullDocDataset, FullDocProcessor
from model import PretrainModel

logger = logging.getLogger(__name__)


def train_loader_fn(config):
    # we pass the plasma_store_address through config
    print(config.plasma_store_address)
    dataset = FullDocDataset(config)
    processor = FullDocProcessor(config)
    # must set rank to ensure correct distributed training
    dataloader = DataDispatcher(local_rank=config.rank, dataset=dataset, processor=processor, num_movers=1, num_workers=4, sorted_output=False,
                            batch_size=config.training.batch_size,plasma_store_address=config.plasma_store_address)
    return dataloader

@hydra.main(config_path="config/config.yaml", strict=False)
def main(config=None):
    # set data loader
    model = PretrainModel(config)
    trainer = Trainer(config=config, train_loader_fn=train_loader_fn, model=model)
    trainer.train()


if __name__ == "__main__":
    main()
