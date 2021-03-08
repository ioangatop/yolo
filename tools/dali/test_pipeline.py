import nvidia.dali.fn as fn
import nvidia.dali as dali
import subprocess
import numpy as np
import cv2
import sys
import os



def setup_dali(
    image_file='/mnt/data/DATASETS/samples/images/image_110.jpg',
    image_dim=[800, 1600],
    batch_size=1,
    num_threads=4,
    device='mixed',
    device_id=0,
    output_dir='./out/',
):

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    pipeline = dali.pipeline.Pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id
    )

    with pipeline:
        data, _ = fn.file_reader(files=[image_file])
        # image preprocess
        images = fn.image_decoder(data, device=device)
        images = fn.resize(images, size=image_dim, mode="not_larger", max_size=image_dim)
        images = fn.pad(images, fill_value=0, shape=[image_dim[0], image_dim[1], 1])
        images = fn.transpose(images, perm=[2, 0, 1])
        images = fn.cast(images, dtype=dali.types.FLOAT)
        images = images / 255.
        # input shape
        input_shape = np.float32((image_dim[0], image_dim[1], 1))
        # original shape
        shapes = fn.peek_image_shape(data)
        shapes = fn.cast(shapes, dtype=dali.types.FLOAT)
        # gather outputs
        out = [
            images,
            input_shape,
            shapes
        ]
        pipeline.set_outputs(*out)

    pipeline.build()
    output = pipeline.run()
    img = output[0].at(0) if device=='cpu' else output[0].as_cpu().at(0)

    img = img.transpose(1, 2, 0)    # HWC
    img = img[:, :, ::-1]           # BGR
    print(img)
    quit()
    cv2.imwrite(os.path.join(output_dir, 'dali_image.jpg'), img)


def main():
    setup_dali()


if __name__ == '__main__':
    main()
