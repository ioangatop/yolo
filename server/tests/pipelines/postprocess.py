import torchvision
import numpy as np
import torch
import time



def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y


def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) \
        - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)


def non_max_suppression(
    prediction,
    conf_thres=0.3,
    iou_thres=0.5,
    merge=False,
    classes=None,
    agnostic=True,
    max_det = 1000
):
    if prediction.dtype is torch.float16:
        prediction = prediction.float()     # to FP32

    nc = prediction[0].shape[1] - 5         # number of classes
    xc = prediction[..., 4] > conf_thres    # candidates

    # Settings
    min_wh, max_wh = 2, 4096    # (pixels) minimum and maximum box width and height
    time_limit = 10.0           # seconds to quit after
    redundant = True            # require redundant detections
    multi_label = nc > 1        # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)         # classes
        boxes, scores = x[:, :4] + c, x[:, 4]               # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:                            # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):                         # Merge NMS (boxes merged using weighted mean)
            try:                                            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]                # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]                   # require redundancy
            except:                                         # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break   # time limit exceeded

    return output


def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
    coords[:, :4] /= gain
    # clip coords
    boxes[:, 0].clamp_(0, img0_shape[1])
    boxes[:, 1].clamp_(0, img0_shape[0])
    boxes[:, 2].clamp_(0, img0_shape[1])
    boxes[:, 3].clamp_(0, img0_shape[0])
    return coords


class ProcessDetections:

    def __init__(self, output_names, box_mode='xywh'):
        self.box_mode = box_mode
        self.output_names = output_names



    @staticmethod
    def _filter_response(response, idx):
        return [res[idx] if res is not None else None for res in response]


    @staticmethod
    def _to_cpu(response):
        return [res.detach().cpu().numpy() if res is not None else None 
                for res in response]


    def _unpack_response(self, response):
        input_shape, target_shape, pred = [
            response.as_numpy(name) for name in self.output_names
        ]
        input_shape = input_shape[0][:2].astype(int)
        target_shape = target_shape[0][:2].astype(int)
        pred = torch.from_numpy(pred[0])
        return input_shape, target_shape, pred


    def forward(self, response, threshold=None, iou_threshold=None, class_id=None):
        input_shape, target_shape, pred = self._unpack_response(response)

        pred /= 1000
        pred = non_max_suppression(pred)

        # process detections
        detections = []
        for det in pred:
            if det is not None and len(det):
                # rescale boxes to target_image size
                input_shape = torch.tensor(input_shape)
                target_shape = target_shape + [3]
                det[:, :4] = scale_coords(input_shape, det[:, :4],
                                          target_shape).round()
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
                        'label': 'n/a'  # data_cfg['names'][category_id]
                    }
                    detections.append(detection)
        return detections


    def __call__(self, response, threshold=None, iou_threshold=None, class_id=None):
        return self.forward(response, threshold, iou_threshold, class_id)
