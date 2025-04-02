# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import logging
import re
from contextlib import nullcontext
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis, unpad_input_for_concatenated_sequences

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.sft_chat_dataset_v2 import MultiTurnSFTDataset, DistributedBatchMultiTurnSFTDatasetSampler, collate_fn

from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.tracking import Tracking
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from peft import LoraConfig, TaskType, get_peft_model

from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))


def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj



class BaseSFTTrainer(object):

    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        # build tokenizer first
        local_model_path = copy_local_path_from_hdfs(src=self.config.model.partial_pretrain, verbose=True)
        from verl.utils import hf_tokenizer
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
        # if self.config.data.chat_template is not None:
        # currently already supported
        #     raise ValueError('Apply Chat template from config is not supported yet.')

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, 'ulysses_sequence_parallel_size', 1)
        self.use_remove_padding = getattr(self.config, 'use_remove_padding', False)
        if self.device_mesh.get_rank() == 0:
            print(f'Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}')
            print(f'Using remove padding: {self.use_remove_padding}')

        self._build_dataloader()
        # build model
        self._build_model_optimizer()

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)
    

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f'Normalize batch size by dp {dp_size}')

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0
    def _build_dataloader(self):
        pass
    
    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_local_path_from_hdfs(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage('Before model allocation', logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(config.model_type)

        if self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1:
            print(self.ulysses_device_mesh.get_group("sp").size())
            from verl.models.transformers.sequence_parallel_monkey_patch import apply_sequence_parallel
            apply_sequence_parallel(group_this=self.ulysses_device_mesh.get_group("sp"), sequence_parallel_mode='ulysses')

        # This may be very large
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings)
        ckpt_state = self.load_checkpoint()
        with init_context():
            if ckpt_state is not None:
                self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                    ckpt_state["path"], 
                    config=config, torch_dtype=torch.float32,
                    attn_implementation='flash_attention_2',
                    trust_remote_code=trust_remote_code)
            else:
                                                                                
                self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                                                config=config,
                                                                                torch_dtype=torch.float32,
                                                                                attn_implementation='flash_attention_2',
                                                                                trust_remote_code=trust_remote_code)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get('use_liger', False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model, fused_linear_cross_entropy=False, cross_entropy=False)

            if self.config.model.get('lora_rank', 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none"
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        log_gpu_memory_usage('After model allocation', logger=logger)
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, Qwen2RMSNorm
        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(self.model,
                                                config=self.config.model.fsdp_config.wrap_policy,
                                                is_lora=self.config.model.get('lora_rank', 0) > 0)
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)
        
        self.fsdp_model = FSDP(module=self.model,
                               auto_wrap_policy=auto_wrap_policy,
                               param_init_fn=init_fn,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               device_mesh=self.device_mesh,
                               sync_module_states=True,
                               device_id=torch.cuda.current_device(),
                               cpu_offload=cpu_offload,
                               use_orig_params=False)

        log_gpu_memory_usage('After FSDP wrapping', logger=logger)


        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                     lr=self.config.optim.lr,
                                     betas=self.config.optim.betas,
                                     weight_decay=self.config.optim.weight_decay)

        if ckpt_state is not None:
            self.optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
        # self.load_checkpoint()
        log_gpu_memory_usage('After initialize optimizer', logger=logger)
        next(iter(self.train_dataloader))
        print(next(iter(self.val_dataloader)))
        self.steps_per_epoch = len(self.train_dataloader) #+ 200
        self.total_steps = (10 + self.steps_per_epoch) * self.config.trainer.total_epochs 

        if self.device_mesh.get_rank() == 0:
            print(
                f'Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}'
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=self.total_steps,
                                                            scaled_loss=1.0)
        if ckpt_state is not None:
            self.lr_scheduler.load_state_dict(ckpt_state["scheduler_state_dict"])
        del ckpt_state

    def _compute_loss_and_backward(self, batch, n_item, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        position_ids = batch['position_ids'].cuda()
        # loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
        labels = batch.pop('labels').cuda()#.reshape(-1).cuda()
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        context = self.sharding_manager if use_sp else nullcontext()
        context = nullcontext()
        group_sp = self.ulysses_device_mesh.get_group("sp")
        with context:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if not use_sp:
                    # Standard forward pass without sequence parallel
                    # labels = input_ids[:, 1:].contiguous()
                    output = self.fsdp_model(input_ids=input_ids,
                                             attention_mask=attention_mask.cuda(),
                                             position_ids=position_ids,
                                             use_cache=False)
                    logits = output.logits.float()

                    # shift_logits = logits[..., :-1, :].contiguous()
                    # shift_labels = labels[:, :-1].contiguous()
                    # Flatten the tokens
                    shift_logits = logits.view(-1, self.model.config.vocab_size)
                    shift_labels = labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss.sum() / n_item.to(loss.dtype)
                else:
                    # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                    # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                    # 1. All SP ranks will receive the *SAME* batch
                    # 2. Different SP groups will receive *DIFFERENT* batches
                    # This is implemented by the DistributedSampler
                    with torch.no_grad():
                        batch_size, seqlen = input_ids.shape
                        # Remove padding
                        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                                attention_mask)  # input_ids_rmpad (total_nnz, ...)
                        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                        # Unpad position_ids to align rotary
                        position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                            indices).transpose(0, 1)

                        # Pad and slice inputs for sequence parallelism
                        input_ids_rmpad_sliced, position_ids_rmpad_padded, position_ids_rmpad_wo_padded, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size(group_sp), group=group_sp)
                        # For computing loss
                    # For computing loss
                        # input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                        # input_ids_rmpad_rolled, _, _,_ = ulysses_pad_and_slice_inputs(
                        #     input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size(group_sp), pad_index=-100)
                        # input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)

                        # Roll_v2 
                        # print(labels.shape)
                        labels,_,*_ = unpad_input(labels.unsqueeze(-1),
                                                                attention_mask) 
                        labels = labels.transpose(0, 1)  
                        labels, _, _,_ = ulysses_pad_and_slice_inputs(
                            labels, None, get_ulysses_sequence_parallel_world_size(group_sp), pad_index=-100, group=group_sp)
                       
                    # input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                    # Forward pass
                        # if torch.distributed.get_rank() == 0:
                        #     import ipdb; ipdb.set_trace()
                        indices_q = torch.arange(position_ids_rmpad_wo_padded.size(1), device=position_ids_rmpad_wo_padded.device, dtype=torch.int32)
                        # print(position_ids_rmpad_wo_padded.size())
                        cu_seq_lens = torch.cat(
                            (
                                indices_q[position_ids_rmpad_wo_padded[0] == 0],
                                torch.tensor(position_ids_rmpad_wo_padded[0].size(), device=position_ids_rmpad_wo_padded.device, dtype=torch.int32),
                            )
                        )
                    output = self.fsdp_model(
                        input_ids=input_ids_rmpad_sliced,
                        attention_mask=None,  # Not needed with flash attention varlen
                        position_ids=position_ids_rmpad_padded,
                        cu_seq_lens_q=cu_seq_lens,
                        use_cache=False)
                    # if torch.distributed.get_rank() == 0:
                    #     import ipdb; ipdb.set_trace()
                    # Compute loss locally then aggregate
                    logits_rmpad = output.logits.squeeze(0).float()
                    labels = labels.to(logits_rmpad.device).view(-1,)
                    # print(input_ids_rmpad_rolled.shape, logits_rmpad.shape)
                    # print(logits_rmpad.shape, labels.shape)
                    loss = loss_fct(logits_rmpad, labels).sum()
                    # if torch.distributed.get_rank() == 0:
                    #     import ipdb; ipdb.set_trace()
                    # # Gather and unpad for sequence parallelism
                    # loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size, group=group_sp)
                    # if torch.distributed.get_rank() == 0:
                    #     tokenizer = self.tokenizer
                    #     import ipdb; ipdb.set_trace()

                    # # This is the loss collected from all ulysses ranks
                    # full_loss = pad_input(hidden_states=loss.unsqueeze(-1),
                    #                       indices=indices,
                    #                       batch=batch_size,
                    #                       seqlen=seqlen)
                    # full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                    # full_loss = full_loss.reshape(-1)
                    # loss_mask = loss_mask.to(full_loss.device)
                    # loss = full_loss * loss_mask
                loss = torch.sum(loss) / (n_item.to(loss.dtype) / get_ulysses_sequence_parallel_world_size(group_sp))

            if do_backward:
                loss.backward()
        # print(loss)
        return loss
            

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()
        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)
        self.optimizer.zero_grad()
        log_gpu_memory_usage('After optimizer zero_grad', logger=logger)
        # For logging
        lenghts = torch.sum(batch['loss_mask'][:, :-1], -1).float()
        # min_length = torch.min(lenghts)
        # max_length = torch.max(lenghts)
        # mean_length = torch.mean(lenghts) 
        global_length = batch['lengths'].reshape(-1)
        # print(global_length)
        # global_length=global_length.sum(dim=-1)
        n_item = torch.sum(batch['loss_mask'][:, :-1]).cuda() #/ mean_length.cuda()  
        # print(global_length)
        # for key in batch.keys():
        #     print(key, batch[key].shape)
        # exit(0)
        data =  TensorDict({key: batch[key] for key in batch.keys() if key != 'lengths'}, batch_size=batch['input_ids'].shape[0])
        # print(global_length)
        # print(batch['input_ids'].shape)
        micro_batches = data.split(global_length.detach().cpu().numpy().tolist())
        # if torch.distributed.get_rank() == 0:
        #     print(lenghts, global_length)
        # micro_batches = batch.split(global_length)
        # for batch in micro_batches:
            # print(batch['input_ids'].shape)
        # exit(0)
        step_loss = 0
        total_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch, n_item=n_item, do_backward=True)
            total_loss+= loss.detach().cpu()
        step_loss = total_loss.item()
        # We don't scale step_loss / n_microbatches here.
        # print("grad_norm")
        grad_norm=self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        log_gpu_memory_usage('Before optimizer step', logger=logger)
        
        self.optimizer.step()

        log_gpu_memory_usage('After optimizer step', logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage('After offload weights', logger=logger)

        # step_loss = total_loss.detach()
        step_loss = torch.tensor(step_loss, device='cuda')
        # sync loss across dp ranks
        torch.distributed.all_reduce(grad_norm, op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        
        return {'train/loss': step_loss.detach().item(), 'train/lr': lr, "gradient_norm": grad_norm.detach().cpu().item()}

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        n_item = torch.sum(batch['loss_mask'].cuda())
        global_length = batch['lengths'].reshape(-1, self.config.data.micro_batch_size_per_gpu)
        # print(global_length)
        global_length=global_length.sum(dim=-1)

        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False, n_item=n_item)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss


    def load_checkpoint(self):
        import glob 
        list_of_checkpoints = glob.glob(os.path.join(self.config.trainer.default_local_dir, 'global_step_*'))
        list_of_checkpoints = sorted(list_of_checkpoints, key=lambda x: int(os.path.basename(x).split('_')[-1]))
        if len(list_of_checkpoints) > 0:
            global_step_before = int(os.path.basename(list_of_checkpoints[-1]).split('_')[-1])
        else:
            global_step_before = 0
        
        model_path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{global_step_before}')
        if os.path.exists(model_path):
            from torch.distributed.fsdp import FullStateDictConfig, StateDictType
            cfg = FullStateDictConfig(offload_to_cpu=True)
            with nullcontext():
                # self.fsdp_model.load_pretrained(model_path)
                automodel = AutoModelForCausalLM.from_pretrained(model_path)
                return {
                    "path": model_path,
                    "state_dict": automodel.state_dict(),
                    "optimizer_state_dict": torch.load(os.path.join(model_path, 'optimizer_state_dict.pth'), map_location="cpu"),
                    "scheduler_state_dict": torch.load(os.path.join(model_path, 'lr_scheduler_state_dict.pth'), map_location="cpu"),
                    "global_step": global_step_before,
                    # "epoch": global_step_before // self.steps_per_epoch,
                }
        return None

    def save_checkpoint(self, step, epoch, step_in_epoch):
        # save checkpoint
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()
            optimizer_state_dict = self.optimizer.state_dict()
            lr_scheduler_state_dict = self.lr_scheduler.state_dict()
        path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            try:
                torch.save(optimizer_state_dict, os.path.join(path, 'optimizer_state_dict.pth'))
            except:
                print("Failed to save optimizer state dict")
            with open(os.path.join(path, 'total_steps.txt'), 'w') as f:
                f.write(str(step))
                f.write(f'\n{epoch}')
                f.write(f'\n{step_in_epoch}')
            try:
                torch.save(lr_scheduler_state_dict, os.path.join(path, 'lr_scheduler_state_dict.pth'))
            except:
                print("Failed to save lr_scheduler state dict")
            # self.optimizer.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger)

        # path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
        import glob 
        list_of_checkpoints = glob.glob(os.path.join(self.config.trainer.default_local_dir, 'global_step_*'))
        list_of_checkpoints = sorted(list_of_checkpoints, key=lambda x: int(os.path.basename(x).split('_')[-1]))
        if len(list_of_checkpoints) > 0:
            global_step_before = int(os.path.basename(list_of_checkpoints[-1]).split('_')[-1])
        else:
            global_step_before = 0
        epoch_before = 0
        steps_in_epoch_before = 0
        global_step = global_step_before
        print(f'Loading checkpoint from {global_step_before}')
        # self.load_checkpoint()
        # global_step_before = 0
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = (len(self.train_dataloader)  + 10)* self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')
        
        # TODO (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.
        torch.distributed.barrier()
        for epoch in range(epoch_before, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            current_step_in_epoch = 0
            print(f'Epoch {epoch+1}/{self.config.trainer.total_epochs}')
            self.steps_per_epoch = len(self.train_dataloader)
            for data in tqdm(self.train_dataloader,
                             total=self.steps_per_epoch,
                             desc=f"Epoch {epoch+1}/{self.config.trainer.total_epochs}"):
                
                # data = TensorDict(data, batch_size=data['input_ids'].shape[0]).cuda()
                global_step += 1
                current_step_in_epoch+=1
                if current_step_in_epoch < steps_in_epoch_before:
                    continue
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)
                
                # for early exit validation
                if global_step >= self.total_training_steps or global_step % 500 == 0:
                    # Perform final validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        # val_data = TensorDict(val_data, batch_size=val_data['input_ids'].shape[0]).cuda()
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        avg_val_loss = torch.mean(torch.stack(val_losses))
                        metric = {'val/loss': avg_val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                    torch.distributed.barrier()
                    # Save final checkpoint
                    self.save_checkpoint(step=global_step, epoch=epoch, step_in_epoch=current_step_in_epoch)
                    # return
                    if global_step >= self.total_training_steps:
                        return

            # validation
                
            val_losses = []
            for data in self.val_dataloader:
                # data = TensorDict(data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
                val_loss = self.validation_step(data)
                val_losses.append(val_loss)
            if rank == 0:
                val_loss = torch.mean(torch.stack(val_losses))
                metric = {'val/loss': val_loss.detach().item()}
                tracking.log(data=metric, step=global_step)
            torch.distributed.barrier()

            # save checkpoint
            self.save_checkpoint(step=global_step, step_in_epoch=0, epoch=epoch + 1)


class FSDPSFTTrainer(BaseSFTTrainer):

    def _build_dataloader(self, device_mesh=None):
        config = self.config
        dataset_class = MultiTurnSFTDataset if config.data.get('use_multiturn', False) else SFTDataset

        if dataset_class == MultiTurnSFTDataset:
            
            self.train_dataset = dataset_class(
                parquet_files=config.data.train_files,
                tokenizer=self.tokenizer,
                messages_key=config.data.messages_key,
                max_length=config.data.max_length,
                cut_of_length=config.data.cut_of_length,
                truncation=config.data.truncation,
                is_hf_dataset=config.data.is_hf_dataset,
                key=config.data.get("key_train", "train"),
                device_mesh=device_mesh,
                cache_path=config.data.cache_path
            )
            self.val_dataset = dataset_class(
                parquet_files=config.data.val_files,
                tokenizer=self.tokenizer,
                messages_key=config.data.messages_key,
                max_length=config.data.max_length,
                cut_of_length=config.data.cut_of_length,
                truncation=config.data.truncation,
                is_hf_dataset=config.data.is_hf_dataset,
                key=config.data.get("key_val", "test"),
                device_mesh=device_mesh,
                cache_path=config.data.cache_path
            )
        else:
        # build dataset
            self.train_dataset = SFTDataset(parquet_files=config.data.train_files,
                                            tokenizer=self.tokenizer,
                                            prompt_key=config.data.prompt_key,
                                            prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                            response_key=config.data.response_key,
                                            response_dict_keys=config.data.get('response_dict_keys', None),
                                            max_length=config.data.max_length,
                                            truncation=config.data.truncation,
                                            device_mesh=device_mesh)
            self.val_dataset = SFTDataset(parquet_files=config.data.val_files,
                                        tokenizer=self.tokenizer,
                                        prompt_key=config.data.prompt_key,
                                        prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                        response_key=config.data.response_key,
                                        response_dict_keys=config.data.get('response_dict_keys', None),
                                        max_length=config.data.max_length,
                                        truncation=config.data.truncation,
                                        device_mesh=device_mesh)

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank('dp')
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f'Using SP rank {rank} and size {world_size} for data distribution')
                print(f'Each SP rank gets different data, but the same data WITHIN the same rank')
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Using FSDP rank {rank} and size {world_size} for data distribution')

        if config.data.get("use_real_packing", True) or config.data.get('use_multiturn', False):
            from functools import partial
            self.train_sampler = DistributedBatchMultiTurnSFTDatasetSampler(
                self.train_dataset,
                shuffle=True,
                num_replicas=world_size,
                rank=rank,
                drop_last=False,
                max_length=config.data.max_length,
                batch_size=config.data.train_batch_size
            )
            
            self.train_dataloader = DataLoader(
                                            dataset=self.train_dataset,
                                            batch_size=config.data.train_batch_size,
                                            sampler=self.train_sampler,
                                            collate_fn=partial(collate_fn, micro_batch_size_per_gpu=config.data.micro_batch_size_per_gpu),
                                            num_workers=32
            )
            self.val_sampler = DistributedBatchMultiTurnSFTDatasetSampler(
                self.val_dataset,
                shuffle=False,
                num_replicas=world_size,
                rank=rank,
                drop_last=False,
                max_length=config.data.max_length,
                batch_size=config.data.train_batch_size
            )
            self.val_dataloader = DataLoader(
                                            dataset=self.val_dataset,
                                            batch_size=config.data.micro_batch_size_per_gpu,
                                            sampler=self.val_sampler,
                                            collate_fn=partial(collate_fn, micro_batch_size_per_gpu=config.data.micro_batch_size_per_gpu),
                                            num_workers=32
            )
        else:
            self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
        
            self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                            batch_size=config.data.train_batch_size,
                                            sampler=self.train_sampler,
                                            num_workers=8,
                                            pin_memory=True,
                                            drop_last=True)

            self.val_sampler = DistributedSampler(self.val_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                drop_last=True)
            self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                            batch_size=config.data.micro_batch_size_per_gpu,
                                            sampler=self.val_sampler,
                                            num_workers=8,
                                            pin_memory=True,
                                            drop_last=True)


from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
import hydra

from torch.distributed.device_mesh import init_device_mesh

from verl.utils.distributed import initialize_global_process_group


@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
                                           mesh_dim_names=('dp', 'sp')) 
    # from torch.distributed import dist
    # print(
    #     dist.get_rank(ulysses_device_mesh.get_group("sp"))
    # )
    # return 0
    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)
    trainer.fit()


if __name__ == '__main__':
    main()
