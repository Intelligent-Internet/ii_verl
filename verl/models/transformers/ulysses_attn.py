# Copyright (c) Microsoft Corporation and Jiarui Fang
# SPDX-License-Identifier: Apache-2.0
# DeepSpeed Team & Jiarui Fang


import torch

from typing import Any
from torch import Tensor
from yunchang.kernels import AttnType, select_flash_attn_impl
import torch.distributed as dist
from yunchang.comm.all_to_all import SeqAllToAll4D
import torch
from transformers.modeling_flash_attention_utils import prepare_fa2_from_position_ids
from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module
from flash_attn import flash_attn_func, flash_attn_varlen_func
import torch.distributed as dist


# def all_to_2d(
#     input: torch.tensor, gather_idx: int = 1, group=None, use_sync: bool = False
# ) -> torch.tensor:
#     """
#     """
#     assert (
#         input.dim() == 2
#     ), f"input must be 2D tensor, got {input.dim()} and shape {input.shape}"

#     seq_world_size = dist.get_world_size(group) 
#     bs, shard_seqlen = input.shape
#     seqlen = shard_seqlen * seq_world_size
#     input_t = (
#             input.reshape(bs, shard_seqlen, seq_world_size)
#         )
#     output = torch.empty_like(input_t)
        

class UlyssesAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
        use_sync (bool): whether to synchronize after all-to-all. This flag can save cuda memory but will slow down the speed.
        attn_type (AttnType): attention type enum
    """

    def __init__(
        self,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_sync: bool = False,
        attn_type : AttnType = AttnType.FA,
    ) -> None:

        super(UlyssesAttention, self).__init__()
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.use_sync = use_sync
        self.attn_type = attn_type

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(device)
        if "Turing" in gpu_name or "Tesla" in gpu_name or "T4" in gpu_name:
            self.attn_type = AttnType.TORCH
        self.attn_fn = select_flash_attn_impl(self.attn_type, stage="fwd-bwd")
        self.logging = False

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        softcap=0.0,
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
        **kwargs: Any
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)
        # import ipdb; 
        # debug on rank 0 only
        
        # scatter 2, gather 1
        if softcap is None:
            softcap = 0.0
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx, self.use_sync)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx, self.use_sync)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx, self.use_sync)
        # position_ids = SeqAllToAll4D.apply(self.spg, position_ids, self.scatter_idx, self.gather_idx, self.use_sync)
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        
        if "position_ids_all_seq" in kwargs:
            # print(q.shape, k.shape, v.shape, 'qkv')
            batch_size = q.shape[0]
            position_ids = kwargs['position_ids_all_seq']
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = (
                prepare_fa2_from_position_ids(q, k, v, position_ids)
            )
            

            cu_seq_lens_q, cu_seq_lens_k = cu_seq_lens
            max_length_q, max_length_k = max_seq_lens
            import torch
            # if torch.distributed.get_rank() == 0:
            #     print(cu_seq_lens_q, cu_seq_lens_k)
            #     import ipdb; ipdb.set_trace()
            #     torch.distributed.barrier()

            context_layer = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seq_lens_q,
                cu_seqlens_k=cu_seq_lens_k,
                max_seqlen_q=max_length_q,
                max_seqlen_k=max_length_k,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
            )
            context_layer = context_layer.view(batch_size, -1, q.shape[-2], q.shape[-1])
        else:
            import torch
            # torch.distributed.barrier()
            # if dist.get_rank() == 0:
            #     print(k.shape)
            #     import ipdb; ipdb.set_trace()
            if 'cu_seq_lens_q' in kwargs:
                # if self.logging==False and torch.distributed.get_rank() == 0:
                #     print("cu_seq_lens_q", kwargs['cu_seq_lens_q'], q.shape, k.shape, v.shape, query.shape, key.shape, value.shape)
                #     self.logging = True
                context_layer = flash_attn_varlen_func(
                    q.squeeze(0),
                    k.squeeze(0),
                    v.squeeze(0),
                    dropout_p=dropout_p,
                    softmax_scale = softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=return_attn_probs,
                    cu_seqlens_q = kwargs['cu_seq_lens_q'],
                    cu_seqlens_k=kwargs['cu_seq_lens_q'],
                    max_seqlen_q=q.size(1),
                    max_seqlen_k=k.size(1),
                    *args
                ) 
                # print(context_layer.shape)
                # context_layer = context_layer.unsqueeze(0)
                
                context_layer = context_layer.view(q.shape[0], -1, q.shape[-2], q.shape[-1])
            else:
                # raise ValueError("No cu_seq_lens_q found in kwargs")
                context_layer = self.attn_fn(
                    q,
                    k,
                    v,
                    dropout_p=dropout_p,
                    softmax_scale = softmax_scale,
                    causal=causal,
                    window_size=window_size,
                    softcap=softcap,
                    alibi_slopes=alibi_slopes,
                    deterministic=deterministic,
                    return_attn_probs=return_attn_probs,
                    *args
            )

        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        # print(context_layer.shape, 'context_layer')
        output = SeqAllToAll4D.apply(
            self.spg, context_layer, self.gather_idx, self.scatter_idx, self.use_sync
        )

        # out e.g., [s/p::h]
        return output