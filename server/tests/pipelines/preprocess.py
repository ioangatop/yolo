from urllib.parse import urlparse
from torch.nn import functional as F
from flask import abort
from io import BytesIO

import requests
import imghdr
import numpy as np
import magic
import os



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
                abort(400)

        except Exception:
            abort(400)

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
            abort(400)


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


    def load_image_dir(self, dir_path):
        path_generator = (os.path.join(dir_path, f) for f in os.listdir(dir_path)
                          if self.is_image(os.path.join(dir_path, f)))
        img_paths = [dir_path] if os.path.isfile(dir_path) else list(path_generator)
        images = [self.load_image(img) for img in img_paths]
        return images


    def array_from_list(self, arrays):
        lengths = list(map(lambda x, arr=arrays: arr[x].shape[0], [x for x in range(len(arrays))]))
        max_len = max(lengths)
        arrays = list(map(lambda arr, ml=max_len: np.pad(arr, ((0, ml - arr.shape[0]))), arrays))
        for arr in arrays:
            assert arr.shape == arrays[0].shape, "Arrays must have the same shape"
        return np.stack(arrays)


    def load_data(self, input):
        if self.is_url(input):
            data = [self.load_image_url(input)]
        elif self.is_file(input):
            data = [self.load_image_file(input)]
        elif os.path.isdir(input):
            data = self.load_image_dir(input)
        else:
            abort(400)

        data = self.array_from_list(data)
        return data


    def __call__(self, f):
        return self.load_data(f)
