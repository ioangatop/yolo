import onnxruntime
import argparse
import numpy as np
import torch
import cv2
import os

from yolo.utils.visualizer import Visualizer
from yolo.utils.general import non_max_suppression, scale_coords
from yolo.data import load_data_cfg


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx-model', type=str, default='./out/model.onnx')
    parser.add_argument('--input-image', type=str, default='/mnt/data/DATASETS/image-object-detection-datasets/crowd-human/val/273278,12fd4b000cf6622b4.jpg')
    parser.add_argument('--img-size', nargs='+', type=int, default=[896, 1536], help='inference size (pixels)')
    parser.add_argument('--data-cfg', default='./data/crowdhuman-visible_head.yaml', help='data.yaml path')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='Object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--max_det', type=int, default=1000, help='Maximum detections')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output', default='./out')
    args = parser.parse_args()
    return args


def input_preprocess(image_path, size=[896, 1536], half=False, device='cpu'):
    img = cv2.imread(image_path)
    assert img is not None, f'Image Not Found {input}'

    # rescale and pad
    h, w, _ = img.shape
    ih, iw = size
    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(img, (nw, nh))
    image_padded = np.full(shape=[ih, iw, 3], fill_value=255.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.
    img = image_padded

    # to numpy
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = img[np.newaxis, ...].astype(np.float32)
    return img


def main():
    args = get_args()

    data_cfg = load_data_cfg(args.data_cfg)
    visualizer = Visualizer()

    # init session
    sess = onnxruntime.InferenceSession(args.onnx_model)
    sess.get_modelmeta()

    # process input data
    target_image = cv2.imread(args.input_image)
    input_image = input_preprocess(args.input_image)
    dict_input = {sess.get_inputs()[0].name: input_image}

    # inference
    pred = sess.run([], dict_input)[0]
    pred = torch.FloatTensor(pred)
    pred /= 1000

    # non-maximum suppression
    pred = non_max_suppression(
        pred,
        args.conf_thres,
        args.iou_thres,
        classes=args.classes,
        agnostic=args.agnostic_nms,
        max_det=args.max_det
    )

    # process detections
    detections = []
    for det in pred:
        if det is not None and len(det):
            # rescale boxes to target_image size
            det[:, :4] = scale_coords(input_image.shape[2:], det[:, :4],
                                      target_image.shape).round()
            # formulate detections in schema
            for *xyxy, conf, class_id in det:
                xyxy = torch.tensor(xyxy).view(-1).tolist()
                xywh = xyxy[0], xyxy[1], xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                score = int((conf.item() * 10**2) + 0.5) / 10**2
                category_id = int(class_id.item())
                detection = {
                    'bbox': xywh,
                    'score': score,
                    'category_id': category_id,
                    'segmentation': [[]],
                    'label': data_cfg['names'][category_id]
                }
                detections.append(detection)

    # plot detections
    save_as = os.path.join(args.output, os.path.basename(args.input_image))
    visualizer(args.input_image, detections, exclude=['labels', 'scores'], save_as=save_as)



if __name__ == '__main__':
    main()
