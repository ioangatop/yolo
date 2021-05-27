from argparse import ArgumentParser
from onnxsim import simplify

import torch.nn as nn
import torch
import onnx
import os


from yolo.layers.activations import Mish
from yolo.modeling.yolo.yolo import Detect
from yolo.modeling.yolo import common
from yolo.modeling import build_model



def get_args():
    parser = ArgumentParser(description='Model Export')
    parser.add_argument('--format', type=str, default='onnx', choices=['onnx'])
    parser.add_argument('--model-cfg', default='./models/COCO-Detection/yolov4-p5.yaml')    
    parser.add_argument('--model-weights', default='/media/braincreator/bigdata01/MODELS/yolo/weights/CrowdHuman/yolov4-p5.pt')
    parser.add_argument('--num-classes', default=2)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--input-shape', type=int, default=[1280, 1280], choices=[416, 640, 896, 1280, 1536])
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output', type=str, default='./out')
    args = parser.parse_args()
    return args


def setup_model(model):
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()
        if isinstance(m, common.Conv) and isinstance(m.act, common.Mish):
            m.act = Mish()
        if isinstance(m, common.BottleneckCSP) or isinstance(m, common.BottleneckCSP2) \
                or isinstance(m, common.SPPCSP):
            if isinstance(m.bn, nn.SyncBatchNorm):
                bn = nn.BatchNorm2d(m.bn.num_features, eps=m.bn.eps, momentum=m.bn.momentum)
                bn.training = False
                bn._buffers = m.bn._buffers
                bn._non_persistent_buffers_set = set()
                m.bn = bn
            if isinstance(m.act, common.Mish):
                m.act = Mish()
        if isinstance(m, Detect):
            m.forward = m.forward_export

    model.eval()
    model.model[-1].export = True   # set Detect() layer export=True
    return model


def to_onnx(input, model, save_as, verbose=True):
    if verbose:
        print(f'Starting ONNX export with onnx {onnx.__version__}...')

    with torch.no_grad():
        torch.onnx.export(
            model,
            input,
            save_as,
            opset_version=12,
            input_names=['input'],
            output_names=['output']
        )

    # validation checks
    onnx_model = onnx.load(save_as)
    model_simp, check = simplify(onnx_model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, save_as)

    if verbose:
        print(f'ONNX export success, saved as {save_as}')



def to_torchscript(input, model, save_as):
    with torch.no_grad():
        ts = torch.jit.trace(model, input)
    ts.save(save_as)



def export_model(args):
    os.makedirs(args.output, exist_ok=True)
    input = torch.zeros((args.batch_size, 3, *args.input_shape), requires_grad=False).to(args.device)

    print(' - Loading model...')
    model = build_model(args.model_cfg, args.model_weights, nc=args.num_classes, eval=True, device=args.device)
    model = setup_model(model)
    model = model.to(args.device)
    model(input)
    print('   Model loaded successfully!')

    if args.format == 'onnx':
        save_as = os.path.join(args.output, 'model.onnx')
        to_onnx(input, model, save_as)

    elif args.format == 'torchscript':
        save_as = os.path.join(args.output, 'model.pt')
        to_torchscript(input, model, save_as)

    else:
        pass


if __name__ == "__main__":
    args = get_args()
    export_model(args)
