import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from torchfly.training.data import DataDispatcher, DataProcessor
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

from typing import Iterator, Tuple

# DATA_DIR = "/home/wuqy1203/Desktop/Projects/LargePretrain/data"

class FullDocDataset(IterableDataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data = []
        self.total_num_sectors = config.data.total_num_sectors
        self.sector_size = self.total_num_sectors // config.training.num_gpus_per_node
        self.data_dir = config.data.data_dir

    def __iter__(self):
        # rank = torch.distributed.get_rank()
        rank = self.config.rank

        while True:
            for i in range(self.sector_size):
                filename = f"{self.data_dir}/{i + rank * self.sector_size}.pkl"
                data = torch.load(filename)
                print(f"{i + rank * self.sector_size}.pkl loaded")
                for item in data:
                    yield item.tolist()

    def __len__(self):
        return len(self.data)

class FullDocProcessor(DataProcessor):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.buffer = []
        self.max_seq_len = config.data.max_seq_len
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.sep_token_id = self.tokenizer.sep_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.mask_token_id = self.tokenizer.mask_token_id
        

    def process(self, item) -> Iterator:
        if len(self.buffer) > 0 and len(self.buffer) < self.max_seq_len - 2:
            self.buffer.append(self.sep_token_id)

        for token_id in item:
            if len(self.buffer) == self.max_seq_len - 2:
                self.buffer = [self.cls_token_id] + self.buffer + [self.sep_token_id]
                yield self.buffer
                self.buffer = []
            self.buffer.append(token_id)

    def collate_fn(self, batch):
        batch = [torch.LongTensor(item) for item in batch]
        batch_input_ids = batch
        batch_input_ids = pad_sequence(batch_input_ids, batch_first=True, padding_value=self.pad_token_id)
        batch_input_ids = batch_input_ids.long()

        batch_input_ids, labels = self.mask_tokens(batch_input_ids)

        batch = {"input_ids": batch_input_ids, "labels": labels}

        return batch

    def mask_tokens(self, inputs: torch.Tensor, mlm_probability: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
            )

        labels = inputs.clone()
        # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
        probability_matrix = torch.full(labels.shape, mlm_probability)
        # special_tokens_mask = [
        #     self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        # ]
        # probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        if self.pad_token_id is not None:
            padding_mask = labels.eq(self.pad_token_id)
            probability_matrix.masked_fill_(padding_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(self.config.model.vocab_size, labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
