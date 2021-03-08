import argparse
import os

from pipelines import TRTInferenceModule, ProcessDetections, ImageLoader, Visualizer



def get_args():
    parser = argparse.ArgumentParser(description="I/O Client Server API")
    parser.add_argument('--image', type=str, help='Path or URL of input image.')
    parser.add_argument('--triton-host', type=str, default='192.168.1.113', help='API triton host')
    parser.add_argument('--triton-port', type=str, default='8000', help='API triton port')
    parser.add_argument('--triton-model-name', type=str, default='detector', help='Triton model name')
    parser.add_argument('--triton-model-version', type=str, default='1', help='Triton model version')
    parser.add_argument('--threshold', type=float, default=None, help='Detection threshold')
    parser.add_argument('--iou-threshold', type=float, default=None, help='IoU threshold for NMS')
    parser.add_argument('--class-id', type=int, default=None, help='Return detections only for class_id')
    parser.add_argument('--output-dir', type=str, default='./out', help='Directory to save the images with drawn detection')
    args = parser.parse_args()
    return args


def setup_pipelines(args):
    dataloader = ImageLoader()
    model_client = TRTInferenceModule(args.triton_host, args.triton_port, 
                                      args.triton_model_name, args.triton_model_version)
    process_responses = ProcessDetections(model_client.output_names)
    return dataloader, model_client, process_responses


def infer(f, dataloader, model_client, process_responses,
          threshold, iou_threshold, class_id):
    inputs = dataloader(f)
    response = model_client(inputs)
    detections = process_responses(response, threshold,
                                   iou_threshold, class_id)
    return detections


def detect(args, dataloader, model_client, process_responses):
    detections = infer(
        args.image,
        dataloader,
        model_client,
        process_responses,
        threshold=args.threshold,
        iou_threshold=args.iou_threshold,
        class_id=args.class_id
    )

    visualizer = Visualizer()
    visualizer(args.image,
               detections,
               os.path.join(args.output_dir, os.path.basename(args.image)))


def test(args, dataloader, model_client, process_responses):

    valid_data = [
        'https://farm4.staticflickr.com/3491/4023151748_cb042e1794_z.jpg',  # url image
        'https://images.wsj.net/im-194039',                                 # url image
        'https://farmhand.ie/wp-content/uploads/2016/04/35-1024x683.jpg'    # url image
    ]

    invalid_data = [
        'https://data.whicdn.com/images/220912012/original.gif',            # gif image
        'https://i.imgur.com/rXB9b3Y.gif',                                  # gif image
        '/dev/null'                                                         # empty file
    ]


    visualizer = Visualizer()

    print('Test of valid data started ...')
    for i,f in enumerate(valid_data):
        detections = infer(f, dataloader, model_client, process_responses,
                           args.threshold, args.iou_threshold, args.class_id)
        visualizer(f,
                   detections,
                   os.path.join(args.output_dir, f'image_{i}.jpg'))
        print(f'- {i+1}/{len(valid_data)} sample done ...')
    print('Test of valid data ended.')


    print('Test of invalid data started ...')
    for i,f in enumerate(invalid_data):
        try:
            detections = infer(f, dataloader, model_client, process_responses,
                            args.threshold, args.iou_threshold, args.class_id,
                            args.poly_points)
        except:
            print(f'- {i+1}/{len(invalid_data)} sample done ...')
    print('Test of invalid data ended.')


def main():
    args = get_args()
    pipelines = setup_pipelines(args)
    if args.image is None:
        test(args, *pipelines)
    else:
        detect(args, *pipelines)


if __name__ == '__main__':
    main()
