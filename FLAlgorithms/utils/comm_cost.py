import os
import torch

def get_full_model_size(model):
    if isinstance(model,dict):
        total_params = sum(p.numel() for p in model.values())
    elif isinstance(model,torch.Tensor):
        total_params =  model.numel()
    else:
        total_params = sum(p.numel() for p in model.parameters())

    size_bytes = total_params * 4
    size_MB = size_bytes / (1024 * 1024)
    return size_MB


def get_quant_model_size(model):
    if isinstance(model,dict):
        total_params = sum(p.numel() for p in model.values())
    elif isinstance(model,torch.Tensor):
        total_params = model.numel()
    else:
        total_params = sum(p.numel() for p in model.parameters())
    size_bytes = total_params
    size_MB = size_bytes / (8 * 1024 * 1024)
    return size_MB
