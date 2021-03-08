from urllib.parse import urlparse
from io import BytesIO

import requests
import numpy as np
import magic
import os


from core.messages import NO_VALID_INPUT


class ImageLoader:

    def __init__(self):
        self.types = ['jpg', 'jpeg', 'bmp',
                      'png', 'tiff', 'pnm']

        self.magic = magic.Magic(mime=True)
        self.header_types = [f'image/{tp}' for tp in self.types]


    def download_image(self, url):
        try:
            r = requests.get(url, stream=True)
            if r.headers['Content-Type'] not in self.header_types:
                raise ValueError(NO_VALID_INPUT.format(input))

        except Exception:
            raise ValueError(NO_VALID_INPUT.format(input))

        bytestream = r.content
        return bytestream


    @staticmethod
    def is_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False


    @staticmethod
    def is_file(file):
        return file is not None and os.path.exists(file) \
               and os.stat(file).st_size != 0


    def check_file_type(self, bytestream):
        file_type = self.magic.from_buffer(bytestream)
        if file_type not in self.header_types:
            raise ValueError(NO_VALID_INPUT.format(input))


    def load_image_url(self, url):
        bytestream = self.download_image(url)
        img = BytesIO(bytestream).read()
        self.check_file_type(img)
        return np.array(list(img)).astype(np.uint8)


    def load_image_file(self, f):
        with open(f, "rb") as file:
            img = file.read()
            self.check_file_type(img)
            return np.array(list(img)).astype(np.uint8)


    def array_from_list(self, arrays):
        lengths = list(map(lambda x, arr=arrays: arr[x].shape[0], [x for x in range(len(arrays))]))
        max_len = max(lengths)
        arrays = list(map(lambda arr, ml=max_len: np.pad(arr, ((0, ml - arr.shape[0]))), arrays))
        for arr in arrays:
            assert arr.shape == arrays[0].shape, "Arrays must have the same shape"
        return np.stack(arrays)


    def load_image(self, input):
        if self.is_url(input):
            image = [self.load_image_url(input)]
        elif self.is_file(input):
            image = [self.load_image_file(input)]
        else:
            raise ValueError(NO_VALID_INPUT.format(input))

        image = self.array_from_list(image)
        return image


    def __call__(self, f):
        return self.load_image(f)
