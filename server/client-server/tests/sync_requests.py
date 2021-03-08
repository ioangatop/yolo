import grequests as asynch

import argparse
import requests
import time


def get_args():
    parser = argparse.ArgumentParser(description="Inference Server")
    parser.add_argument('--host', type=str, default='192.168.1.113', help='API triton host')
    parser.add_argument('--port', type=str, default='8080', help='API triton port')
    parser.add_argument('--threshold', type=float, default=-1, help='Detection threshold')
    parser.add_argument('--iou-threshold', type=float, default=-1, help='IoU threshold for NMS')
    parser.add_argument('--class-id', type=int, default=0, help='Return detections only for class_id')
    parser.add_argument('--poly-points', type=int, default=150, help='Upper bound of numpers of points that make the polygon')
    parser.add_argument('--output-dir', type=str, default='./out', help='Directory to save the images with drawn detection')
    args = parser.parse_args()
    return args


def url_to_request(url, host, port):
    return f'http://{host}:{port}/api/models/infer?url={url}&class_id=0'


def get_detections(responce):
    data = responce.json()
    try:
        detections = data['detections']
    except:
        print(data)
        quit()
    return detections


def test(args):

    urls = [
        'https://farm4.staticflickr.com/3491/4023151748_cb042e1794_z.jpg',
        'https://farmhand.ie/wp-content/uploads/2016/04/35-1024x683.jpg',
        'https://upload.wikimedia.org/wikipedia/commons/c/c0/See_armisvesi_in_finnland.JPG',
        'https://images.pexels.com/photos/804463/pexels-photo-804463.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500',   # BUG results are not print on visualiser
        'https://www.holidify.com/images/cmsuploads/compressed/800px-Hong_Kong_International_Airport_duty-free_shops_01_20200123112249.JPG',
        'https://pbs.twimg.com/media/D9lWa1rUYAA_N-u.png',
        'https://pbs.twimg.com/media/CJexA6hWIAAgr_F.jpg:large',
        'https://upload.wikimedia.org/wikipedia/commons/9/94/TIFF_Event_Pecaut_02.jpg',
        'https://res-1.cloudinary.com/hc2kqirjb/image/upload/c_fill,h_1200,q_45,w_2000/v1478680842/ftkx9c8lq7wayhjl785y.png',
        'https://www.access-is.com/storage/uploads/industry/airports-airlines/Airport-Boarding-Gate-Article1-t_gflyu.jpg'
    ]
    http_requests = [url_to_request(url, args.host, args.port) for url in urls]

    print(f'>> Inference started... \n   - Number of requests: {len(http_requests)}')
    tic = time.time()
    for rq in http_requests:
        response = requests.get(rq)
        detections = get_detections(response)
    tac = round(time.time() - tic, 2)
    print(f'   Inference ended. (Total ~ {tac} sec | {round(len(http_requests) / tac, 2)} request per sec)\n')


def main():
    args = get_args()
    test(args)


if __name__ == '__main__':
    main()
