import os
from typing import List

import numpy as np
import torch


def sync_processes():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def is_main_process():
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    else:
        return True


def all_gather_list(input_list: List):
    output_list = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(
        output_list,
        input_list,
    )
    return [entry for output in output_list for entry in output]


def device():
    if torch.distributed.is_initialized():
        return torch.device("cuda:%d" % torch.distributed.get_rank())
    if torch.cuda.is_available() and os.environ.get("CUDA_DEVICE") is not None:
        return torch.device("cuda:%d" % os.eviron["CUDA_DEVICE"])
    return (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def to_device(d, device_=None):
    if device_ is None:
        device_ = device()
    for k in d:
        if type(d[k]) == dict:
            d[k] = to_device(d[k], device_)
        elif type(d[k]) == torch.Tensor:
            d[k] = d[k].to(device_)
    return d


def to_list(tensor):
    return tensor.detach().cpu().tolist()


_G = "Ġ"


def special_token_ids(tokenizer):
    return set(
        {
            tokenizer.bos_token,
            tokenizer.eos_token,
            tokenizer.sep_token,
            tokenizer.cls_token,
            tokenizer.pad_token,
            tokenizer.mask_token,
        }
    )


def concatenate_pair(inputs):
    out = {}
    for k in ("input_ids", "attention_mask"):
        out[k] = torch.cat([inputs["a"][k], inputs["b"][k]], dim=-1)
    B = out["input_ids"].shape[0]
    device = out["input_ids"].device
    a_positions = (
        torch.arange(inputs["a"]["input_ids"].shape[-1], device=device)
        .unsqueeze(0)
        .expand(B, -1)
    )
    b_positions = (
        torch.arange(inputs["b"]["input_ids"].shape[-1], device=device)
        .unsqueeze(0)
        .expand(B, -1)
    ) + inputs["a"]["length"].unsqueeze(-1)
    out["position_ids"] = torch.cat([a_positions, b_positions], dim=-1)
    out["label"] = inputs["label"]
    out["uid"] = inputs["uid"]
    return out
