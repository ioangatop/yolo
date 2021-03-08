import argparse
import requests
import os

from utils import Visualizer
from flask import abort



def get_args():
    parser = argparse.ArgumentParser(description="Inference Server")
    parser.add_argument('--host', type=str, default='0.0.0.0', help='API triton host')
    parser.add_argument('--port', type=str, default='8000', help='API triton port')
    parser.add_argument('--threshold', type=float, default=-1, help='Detection threshold')
    parser.add_argument('--iou-threshold', type=float, default=-1, help='IoU threshold for NMS')
    parser.add_argument('--class-id', type=int, default=-1, help='Return detections only for class_id')
    parser.add_argument('--poly-points', type=int, default=150, help='Upper bound of numpers of points that make the polygon')
    parser.add_argument('--output-dir', type=str, default='./out', help='Directory to save the images with drawn detection')
    args = parser.parse_args()
    return args


def infer(f, host, port, threshold, iou_threshold, class_id, poly_points): 

    r = requests.get(
        url=f'http://{host}:{port}/api/infer?',
        params = {'url':f,
                  'threshold': threshold,
                  'iou_threshold': iou_threshold,
                  'class_id': class_id,
                  'poly_points': poly_points
                  }) 

    data = r.json()
    detections = data['detections']
    return detections


def test(args):

    data = [
        'https://farm4.staticflickr.com/3491/4023151748_cb042e1794_z.jpg',
        'https://farmhand.ie/wp-content/uploads/2016/04/35-1024x683.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/c/c0/See_armisvesi_in_finnland.JPG',
        'https://images.pexels.com/photos/804463/pexels-photo-804463.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500',
        'https://www.holidify.com/images/cmsuploads/compressed/800px-Hong_Kong_International_Airport_duty-free_shops_01_20200123112249.JPG',
        'https://pbs.twimg.com/media/D9lWa1rUYAA_N-u.png',
        'https://pbs.twimg.com/media/CJexA6hWIAAgr_F.jpg:large',
        'https://upload.wikimedia.org/wikipedia/commons/9/94/TIFF_Event_Pecaut_02.jpg',
        'https://res-1.cloudinary.com/hc2kqirjb/image/upload/c_fill,h_1200,q_45,w_2000/v1478680842/ftkx9c8lq7wayhjl785y.png'
    ]

    visualizer = Visualizer()

    print('Test started ...')
    for i,f in enumerate(data):

        detections = infer(f, args.server_url, args.threshold,
                           args.iou_threshold, args.class_id, args.poly_points)

        visualizer(f,
                   detections,
                   os.path.join(args.output_dir, f'image_{i}.jpg'))

        print(f'- {i+1}/{len(data)} sample done ...')

    print('Test ended.')


def main():
    args = get_args()

    test(args)


if __name__ == '__main__':
    main()
