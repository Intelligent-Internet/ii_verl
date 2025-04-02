# modified from
# 1. https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/adapters/hf_adapter.py
# 2. https://github.com/jzhang38/EasyContext/
from functools import partial
import torch
import torch.distributed as dist
import transformers
import transformers.modeling_flash_attention_utils
from transformers.utils import LossKwargs
# from ring_flash_attn import zigzag_ring_flash_attn_func
# from yunchang import UlyssesAttention
from typing import Optional, TypedDict
import transformers
from yunchang.kernels import AttnType
from verl.models.transformers.ulysses_attn import UlyssesAttention

from typing import Optional, Tuple

import torch

from transformers.utils import is_flash_attn_greater_or_equal_2_10


_use_top_left_mask = not is_flash_attn_greater_or_equal_2_10()


def interface_hf(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    # This is before the transpose
    seq_len = query.shape[2]

    # FA2 uses non-transposed inputs
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (usually our RMSNorm modules handle it correctly)
    target_dtype = None
    if query.dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(module.config, "_pre_quantization_dtype"):
            target_dtype = module.config._pre_quantization_dtype
        else:
            target_dtype = next(layer for layer in module.modules() if isinstance(layer, torch.nn.Linear)).weight.dtype

    # FA2 always relies on the value set in the module, so remove it if present in kwargs to avoid passing it twice
    kwargs.pop("is_causal", None)

    attn_output = transformers.modeling_flash_attention_utils._flash_attention_forward(
        query,
        key,
        value,
        attention_mask,
        query_length=seq_len,
        is_causal=module.is_causal,
        dropout=dropout,
        softmax_scale=scaling,
        sliding_window=sliding_window,
        softcap=softcap,
        use_top_left_mask=_use_top_left_mask,
        target_dtype=target_dtype,
        **kwargs,
    )

    return attn_output, None

def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0,
    deterministic=False,
    sliding_window=None,
    is_causal=True,
    group=None,
    mode="ulysses",
    **kwargs,
):
    if mode == "zigzag-ring":
        raise NotImplementedError("Zigzag-ring is not supported for Ulysses.")
        attn_output = zigzag_ring_flash_attn_func(
            query_states, key_states, value_states, dropout, deterministic=deterministic, causal=is_causal, group=group
        )
    elif mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_type=AttnType.FA)
        attn_output = dist_attn(query_states, key_states, value_states, deterministic=deterministic, dropout_p=dropout, causal=is_causal, **kwargs)
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output





def apply_sequence_parallel(group_this, sequence_parallel_mode='ulysses', full_determinism=False):
    if group_this is None:
        print("No sequence parallelism")
        return None  # no sequence parallelism
    print("apply sequence parallel")
    assert isinstance(group_this, torch.distributed.ProcessGroup) 
    try:
        # old_flash_attention_forward = transformers.modeling_flash_attention_utils._flash_attention_forward
        if sequence_parallel_mode == "zigzag-ring":
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this, mode=sequence_parallel_mode, deterministic=full_determinism)
            # assert check_params(old_flash_attention_forward, new_flash_attention_forward)
        elif sequence_parallel_mode == "ulysses":
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this, mode=sequence_parallel_mode, deterministic=full_determinism)
        else:
            raise NotImplementedError("Other sequence parallel modes are to be implemented.")

        # monkey patching
        transformers.modeling_flash_attention_utils._flash_attention_forward = new_flash_attention_forward
        # 
        from transformers.integrations.flash_attention import flash_attention_forward
        transformers.modeling_utils.ALL_ATTENTION_FUNCTIONS['flash_attention_2'] = interface_hf
        transformers.modeling_flash_attention_utils.FlashAttentionKwargs = FlashAttentionKwargs
        from importlib import reload  # Python 3.4+
        
        transformers.models.qwen2.modeling_qwen2.KwargsForCausalLM = KwargsForCausalLM
        reload(
            transformers.models.qwen2.modeling_qwen2
        )
        

    except Exception:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
        )

    return group_this




class FlashAttentionKwargs(TypedDict, total=False):
    """
    Keyword arguments for Flash Attention with Compile.

    Attributes:
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    cu_seq_lens_q: Optional[torch.LongTensor]
    cu_seq_lens_k: Optional[torch.LongTensor]
    max_length_q: Optional[int]
    max_length_k: Optional[int]
    position_ids_all_seq: Optional[torch.LongTensor]


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...
