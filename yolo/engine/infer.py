from argparse import ArgumentParser
from tqdm import tqdm
import torch
import os

from yolo.utils.torch_utils import select_device
from yolo.utils.visualizer import Visualizer
from yolo.data.datasets import LoadImages
from yolo.utils.general import check_img_size, scale_coords, non_max_suppression
from yolo.modeling import build_model
from yolo.data import load_data_cfg



def get_args():
    parser = ArgumentParser(description='Inference Engine')
    parser.add_argument('--input', default='/mnt/data/DATASETS/samples/images')
    parser.add_argument('--base-model', default=None)
    parser.add_argument('--data-cfg', default='./data/crowdhuman-visible_head.yaml', help='data.yaml path')
    parser.add_argument('--model-cfg', default='./models/COCO-Detection/yolov4-p5.yaml')    
    parser.add_argument('--model-weights', default='/media/braincreator/bigdata01/MODELS/yolo/weights/CrowdHuman/yolov4-p5.pt')
    parser.add_argument('--input-size', type=int, default=896, choices=[416, 640, 896, 1280, 1536])
    parser.add_argument('--max-det', type=int, default=1000)
    parser.add_argument('--conf-thres', type=float, default=0.25, help='Object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--half', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output', default='./out')
    args = parser.parse_args()
    return args



def check_initialization(model, input_size, device, half=False):
    # check input_size relatively to the model
    check_img_size(input_size, s=model.stride.max())
    # perform a forward pass
    dummy_input = torch.zeros((1, 3, input_size, input_size), device=device)
    _ = model(dummy_input.half() if half else dummy_input) if device.type != 'cpu' else None


@torch.no_grad()
def launch_inferece(args):
    device = select_device(args.device)
    data_cfg = load_data_cfg(args.data_cfg)
    model = build_model(args.model_cfg, args.model_weights, nc=data_cfg['nc'], eval=True, device=device)
    dataloader = LoadImages(args.input, img_size=args.input_size)
    visualizer = Visualizer()

    check_initialization(model, args.input_size, device, args.half)

    for image_path, input_image, target_image, _ in tqdm(dataloader,
                                                         desc='>>',
                                                         total=len(dataloader)
                                                         ):
        # image to tensor
        input_image = torch.from_numpy(input_image).to(device)
        input_image = input_image.half() if args.half else input_image.float()
        input_image /= 255.0
        if input_image.ndimension() == 3:
            input_image = input_image.unsqueeze(0)

        # forward pass
        pred = model(input_image, augment=args.augment)[0]

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
        save_as = os.path.join(args.output, os.path.basename(image_path))
        visualizer(image_path, detections, exclude=['labels', 'scores'], save_as=save_as)


if __name__ == "__main__":
    args = get_args()
    launch_inferece(args)
