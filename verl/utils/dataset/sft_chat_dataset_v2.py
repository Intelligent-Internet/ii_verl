"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

from typing import List, Union
import torch
import math
import torch.distributed as dist
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
from typing import Optional
from datasets import concatenate_datasets


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 messages_key='messages',  # Key for the messages list in the parquet file
                 max_length=1024,
                 truncation='error',
                 is_hf_dataset=False,
                 key="train",
                 device_mesh=None,
                 cut_of_length=1024,
                 cache_path=None):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation
        # for simple support hf dataset
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.messages_key = messages_key
        if is_hf_dataset:
            from datasets import load_dataset 
            if isinstance(parquet_files, str) == True:
                parquet_files = [parquet_files] 
            if cache_path is None:
                hf_datasets = [load_dataset(parquet_file, split=key) for parquet_file in parquet_files] 
            else:
                hf_datasets = [load_dataset(parquet_file, split=key, cache_dir=cache_path) for parquet_file in parquet_files] 
            for i in range(len(hf_datasets)):
                if "CategoryAssignment" in hf_datasets[i].column_names:
                    hf_datasets[i] = hf_datasets[i].filter(
                        lambda x: x['CategoryAssignment'] == "Mathematics & Statistics",
                        num_proc=10
                    )
                hf_datasets[i] = hf_datasets[i].map(
                    self.process_item_row,
                    num_proc=256,
                )
                
                print(f"Filtering {len(hf_datasets[i])} rows")
                hf_datasets[i] = hf_datasets[i].filter(
                    lambda x: len(x['input_ids']) <= cut_of_length,
                    num_proc=32
                )
                print(f"After filtering {len(hf_datasets[i])} rows")
            # hf_datasets[0].save_to_disk("hf_datasets_0")
            # exit(0)
            self.final_dataset = concatenate_datasets(hf_datasets)
        else:
            raise NotImplementedError("Only support hf dataset for now")
        self.lengths = self.final_dataset['lengths']
        self.max_length = max_length


    def __len__(self):
        return len(self.final_dataset)

    def process_item_row(self, row):
        messages = row[self.messages_key]
        if messages[0]['role'] != 'system':
            messages = [{"role": "system", "content": ""}] + messages
        tokenizer = self.tokenizer
        full_tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors='pt', add_generation_prompt=False)
        input_ids = full_tokens[0]  # The output is already a tensor
        attention_mask = torch.ones_like(input_ids)
        loss_mask = torch.zeros_like(input_ids, dtype=torch.long)
        # Process each message to find assistant responses
        current_length = 0
        for i, msg in enumerate(messages):
            # Get tokens for messages up to this point to find the start position
            prefix_messages = messages[:i+1]
            prefix_tokens = tokenizer.apply_chat_template(prefix_messages, tokenize=True, return_tensors='pt', add_generation_prompt=False)
            
            # Get tokens for messages up to previous point
            prev_tokens = tokenizer.apply_chat_template(messages[:i], tokenize=True, return_tensors='pt', add_generation_prompt=False) if i > 0 else None
            
            # Calculate start and end positions
            start_pos = prev_tokens[0].shape[0] if prev_tokens is not None else 0
            end_pos = prefix_tokens[0].shape[0]
            
            # If this is an assistant message, set loss mask
            if msg['role'] == 'assistant':
                loss_mask[start_pos:end_pos] = 1
        # Create position IDs
        # position_ids = torch.arange(len(input_ids), dtype=torch.long)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Zero out position IDs for padding
        position_ids = position_ids * attention_mask
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
            "lengths": len(input_ids)
        }
    

    def __getitem__(self, items):
        
        items = [self.final_dataset[item] for item in items]

        for i in range(len(items)):
            if len(items[i]['input_ids']) <= self.max_length:
                items[i]['input_ids'] = torch.LongTensor(items[i]['input_ids'])
                items[i]['attention_mask'] = torch.LongTensor(items[i]['attention_mask'])
                items[i]['loss_mask'][-1] = 0 # mask out the last token in the response
                items[i]['loss_mask'] = torch.tensor(items[i]['loss_mask'])
                items[i]['labels'] = items[i]['input_ids'][ 1:] 
                items[i]['labels'] = torch.cat([items[i]['labels'], torch.zeros(size=(1,), dtype=items[i]['labels'].dtype) - 100], dim=0)
                items[i]['position_ids'] =torch.tensor(items[i]['position_ids'])
                if len(items[i]['input_ids']) < self.max_length:
                    items[i]['input_ids'] = torch.cat([items[i]['input_ids'], torch.zeros(size=(self.max_length - len(items[i]['input_ids']),), dtype=items[i]['input_ids'].dtype)], dim=0)
                    items[i]['attention_mask'] = torch.cat([items[i]['attention_mask'], torch.zeros(size=(self.max_length - len(items[i]['attention_mask']),), dtype=items[i]['attention_mask'].dtype)], dim=0)
                    items[i]['loss_mask'] = torch.cat([items[i]['loss_mask'], torch.zeros(size=(self.max_length - len(items[i]['loss_mask']),), dtype=items[i]['loss_mask'].dtype)], dim=0)
                    items[i]['labels'] = torch.cat([items[i]['labels'], torch.zeros(size=(self.max_length - len(items[i]['labels']),), dtype=items[i]['labels'].dtype) - 100], dim=0)
                    items[i]['position_ids'] = torch.cat([items[i]['position_ids'], torch.zeros(size=(self.max_length - len(items[i]['position_ids']),), dtype=items[i]['position_ids'].dtype)], dim=0)
                
            else:   
                raise ValueError(f"Input ids length is greater than max length: {len(items[i]['input_ids'])}")
        return items
    
def collate_fn(batch, micro_batch_size_per_gpu):
    input_ids = [] 
    attention_mask = []
    position_ids = []
    loss_mask = []
    lengths = []
    labels = []
    current = []
    # max_length = max([item['input_ids'].shape[0] for item in batch])
    new_batch = []
    for item in batch:
        # lengths.append(len(item))
        current.append(len(item))
        if len(current) == micro_batch_size_per_gpu:
            lengths.append(sum(current))
            current = []
        new_batch.extend(item)
    if len(current) > 0:
        raise ValueError(f"Current batch size is not equal to micro_batch_size_per_gpu: {len(current)}")
    input_ids = torch.stack([item['input_ids'] for item in new_batch], dim=0)
    attention_mask = torch.stack([item['attention_mask'] for item in new_batch], dim=0)
    position_ids = torch.stack([item['position_ids'] for item in new_batch], dim=0)
    loss_mask = torch.stack([item['loss_mask'] for item in new_batch], dim=0)
    labels = torch.stack([item['labels'] for item in new_batch], dim=0)
    # labels = input_ids[:, 1:]
    # labels = torch.cat([labels, torch.zeros(size=( labels.shape[0], 1), dtype=labels.dtype) - 100], dim=1)
    # loss_mask = torch.stack(loss_mask, 0)
    labels = torch.where(loss_mask == 1, labels, -100)
    # print(lengths)
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_ids': position_ids,
        'loss_mask': loss_mask,
        # 'lengths': torch.stack(lengths, dim=0).long(),
        "labels": labels.long(),
        "lengths": torch.LongTensor(lengths)
    }

class DistributedBatchMultiTurnSFTDatasetSampler(Sampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        max_length: int = 1024,
        batch_size: int = 1,
    ) -> None:
        self.max_length = max_length
        self.batch_size = batch_size
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
    
    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
    
    def __len__(self):
        return self.total

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        
        
        # Do sampling Lenghts here
        # First we  split the indices to 32 parts and sort the indices by the lengths and merge them
        import numpy as np
        lengths = np.array(self.dataset.lengths)
        length_part = len(indices) // 32
        length_part = max(length_part, 32)
        indices_parts = [indices[i:i+length_part] for i in range(0, len(indices), length_part)] 
        # indices = []
        # for indices_part in indices_parts:
        #     indices_part = [x for _, x in sorted(zip(lengths[indices_part], indices_part), key=lambda pair: pair[0])]
        #     indices.extend(indices_part)
        
        

        packs_indices = []
        packs = []

        current_pack = []
        current_pack_index = []

        for index in indices:
            length = lengths[index]
            if sum(current_pack) + length <= self.max_length:
                current_pack.append(length)
                current_pack_index.append(index)
            else:
                packs.append(current_pack[:])
                packs_indices.append(current_pack_index[:])
                current_pack = [length]
                current_pack_index = [index]
        if len(current_pack) > 0:
            packs.append(current_pack[:])
            packs_indices.append(current_pack_index[:])
        
        
        batch_size = self.batch_size

        # We always append the last batch for simplicity 
        # Todo: support drop last
        
        while (len(packs_indices)) % self.num_replicas != 0:
            packs_indices.append(packs_indices[0])

        print(f"Packs indices: {len(packs_indices)}")
        packs_indices_rank = []
        for index, pack in enumerate(packs_indices):
            if index % self.num_replicas == self.rank:
                packs_indices_rank.append(pack)

        assert all([isinstance(pack, list) for pack in packs_indices_rank])
        self.total = len(packs_indices_rank)
        for batch_indices in packs_indices_rank:
            yield batch_indices


                    

        

        
        
        

            