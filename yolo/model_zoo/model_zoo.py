import gdown
import torch
import os

from yolo.utils.torch_utils import intersect_dicts


class _ModelZoo:
    """
    Mapping from names to officially released YOLO pre-trained models.
    """

    # url
    MODEL_TO_URL = {
        'yolov4-p5': 'https://drive.google.com/uc?id=1aXZZE999sHMP1gev60XhNChtHPRMH3Fz',
        'yolov4-p6': 'https://drive.google.com/uc?id=1aB7May8oPYzBqbgwYSZHuATPXyxh9xnf',
        'yolov4-p7': 'https://drive.google.com/uc?id=18fGlzgEJTkUEiBG4hW00pyedJKNnYLP3'

    }

    # local
    MODEL_TO_WEIGHT = {
        'yolov4-p5': '/media/braincreator/bigdata01/MODELS/yolo/weights/COCO2017/yolov4-p5.pt',
        'yolov4-p6': '/media/braincreator/bigdata01/MODELS/yolo/weights/COCO2017/yolov4-p6.pt',
        'yolov4-p7': '/media/braincreator/bigdata01/MODELS/yolo/weights/COCO2017/yolov4-p7.pt'

    }


def get_checkpoint_weight(model_name):
    if model_name in _ModelZoo.MODEL_TO_WEIGHT:
        return _ModelZoo.MODEL_TO_WEIGHT[model_name]
    else:
        raise RuntimeError("{} not available in Model Zoo!".format(model_name))


def get_checkpoint_url(model_name):
    if model_name in _ModelZoo.MODEL_TO_URL:
        return _ModelZoo.MODEL_TO_URL[model_name]
    else:
        raise RuntimeError("{} not available in Model Zoo!".format(model_name))


def download_weights(url, output, verbose=False):
    gdown.download(url, output, quiet=verbose) 
