import torch

def create_mask(param, sparsity):
    return (torch.rand_like(param) > sparsity).float()

def initialize_masks(model, sparsity):
    masks = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            masks[name] = create_mask(param, sparsity)
    return masks

def apply_masks(model, masks):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name])