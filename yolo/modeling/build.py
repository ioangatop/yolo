import torch

from yolo.utils.torch_utils import intersect_dicts
from .yolo import YOLO


def load_weights(model, weights, device):
    # NOTE when model was saved unfused
    # state_dict = intersect_dicts(
    #     torch.load(weights),
    #     model.state_dict()
    # )
    # model.load_state_dict(state_dict, strict=False)
    # return model

    # NOTE when model was saved fused
    model = model.float().fuse()
    ckpt = torch.load(weights)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    return model


def build_model(cfg, weights=None, nc=None, eval=False, device='cuda'):
    model = YOLO(cfg, nc=nc)
    if weights:
        model = load_weights(model, weights, device)
    if eval is True:
        model = model.eval()
    model = model.to(device)
    return model
