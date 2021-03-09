import torch

from yolo.utils.torch_utils import intersect_dicts
from .yolo import YOLO


def load_weights(model, weights, fused=False):
    if fused:
        model = model.float().fuse()
        ckpt = torch.load(weights)
        state_dict = ckpt['model'].float().state_dict()
        model.load_state_dict(state_dict, strict=False)
    else:
        state_dict = intersect_dicts(
            torch.load(weights),
            model.state_dict()
        )
        model.load_state_dict(state_dict, strict=False)
    return model


def build_model(cfg, weights=None, nc=None, eval=False, fuse=False, device='cuda'):
    model = YOLO(cfg, nc=nc)
    if weights:
        model = load_weights(model, weights, eval)
    if eval is True:
        model = model.eval()
    if fuse is True:
        model = model.fuse()
    model = model.to(device)
    return model
