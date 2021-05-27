from tqdm import tqdm
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
    parser.add_argument('--image-dir', type=str, default='/mnt/data/DATASETS/samples/human-crowd')
    parser.add_argument('--onnx-model', type=str, default='/workdir/tools/resources/model.onnx')
    parser.add_argument('--img-size', nargs='+', type=int, default=[1280, 1280], help='inference size (pixels)')
    parser.add_argument('--data-cfg', default='/workdir/data/crowdhuman-visible_head.yaml', help='data.yaml path')
    parser.add_argument('--conf-thres', type=float, default=0.2, help='Object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--agnostic-nms', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--max_det', type=int, default=1000, help='Maximum detections')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--output', default='./out')
    args = parser.parse_args()
    return args


class ImageLoader:
    def __init__(self, path, size=[1280, 1280], valid_exts=('.jpg', '.jpeg', '.png'), level=None, contains=None):
        super().__init__()
        self.path = path
        self.size = size
        self.valid_exts = valid_exts
        self.level = level
        self.contains = contains
        self.len = self.init_len(path, valid_exts, level, contains)
        self.cuda = True


    def init_len(self, path, valid_exts=None, level=None, contains=None):
        total = 0
        for (_, _, filenames) in self.walk_to_level(path, level):
            for filename in sorted(filenames):
                if contains is not None and contains not in filename:
                    continue
                ext = filename[filename.rfind("."):].lower()
                if valid_exts and ext.endswith(valid_exts):
                    total += 1
        return total

    @staticmethod
    def walk_to_level(path, level=None):
        if level is None:
            yield from os.walk(path)
            return

        path = path.rstrip(os.path.sep)
        num_sep = path.count(os.path.sep)
        for root, dirs, files in os.walk(path):
            yield root, dirs, files
            num_sep_this = root.count(os.path.sep)
            if num_sep + level <= num_sep_this:
                del dirs[:]

    def list_files(self, path, valid_exts=None, level=None, contains=None):
        for (root_dir, dir_names, filenames) in self.walk_to_level(path, level):
            for filename in sorted(filenames):
                if contains is not None and contains not in filename:
                    continue
                ext = filename[filename.rfind("."):].lower()
                if valid_exts and ext.endswith(valid_exts):
                    file = os.path.join(root_dir, filename)
                    yield file

    def read_image(self, image_path):
        img = cv2.imread(image_path)
        # rescale and pad
        h, w, _ = img.shape
        ih, iw = self.size
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

    def generator(self):
        source = self.list_files(self.path, self.valid_exts, self.level, self.contains)
        while self.has_next():
            try:
                image_file = next(source)
                image = self.read_image(image_file)
                data = {
                    "id": image_file,
                    "image": image
                }
                if self.filter(data):
                    yield self.map(data)
            except StopIteration:
                return

    def __iter__(self):
        return self.generator()

    def __or__(self, other):
        if other is not None:
            other.source = self.generator()
            return other
        else:
            return self

    def filter(self, data):
        return True

    def map(self, data):
        return data

    def has_next(self):
        return True

    def __len__(self):
        return self.len


@torch.no_grad()
def main():
    args = get_args()

    data_cfg = load_data_cfg(args.data_cfg)
    dataloader = ImageLoader(args.image_dir)
    visualizer = Visualizer()

    # init session
    sess = onnxruntime.InferenceSession(args.onnx_model)
    sess.get_modelmeta()


    for data in tqdm(dataloader, desc='>>', total=len(dataloader)):
        image_path, input_image = data['id'], data['image']
        target_image = cv2.imread(image_path)
        dict_input = {sess.get_inputs()[0].name: input_image}

        # inference
        pred = sess.run([], dict_input)[0]
        pred = torch.FloatTensor(pred)

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
                det[:, :4] = scale_coords(input_image.shape[2:], det[:, :4],
                                        target_image.shape).round()

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



if __name__ == '__main__':
    main()
