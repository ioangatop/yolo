import os


class ImageLoader:
    def __init__(self, path, rgb, valid_exts=('.jpg', '.jpeg', '.png'), level=None, contains=None):
        super().__init__()
        self.path = path
        self.rgb = rgb
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

    def read_image(self, img_pth):
        return img_pth

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


root = '/mnt/data/DATASETS/samples/'

dataloader = ImageLoader(root, rgb=None)

for i,img in enumerate(dataloader):
    os.rename(img['id'], os.path.dirname(img['id']) + f'/image_{i}.jpg')
