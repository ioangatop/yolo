import nvidia.dali.fn as fn
import nvidia.dali as dali
import numpy as np
import os



def setup_dali(
    input_name='DALI_INPUT_0',
    image_dim=[896, 1536],
    batch_size=1,
    num_threads=4,
    device='cpu',
    device_id=0,
    output_dir='./out/',
):

    pipeline = dali.pipeline.Pipeline(
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id
    )

    with pipeline:
        data = fn.external_source(name=input_name, device="cpu")
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

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    pipeline.serialize(filename=os.path.join(output_dir, 'model.dali'))


def main():
    setup_dali()


if __name__ == '__main__':
    main()
