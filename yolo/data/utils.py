import random
import yaml


def load_data_cfg(data):
    with open(data) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    if not data.get('colors'):
        data['colors'] = [
            [random.randint(0, 255) for _ in range(3)]
            for _ in range(len(data['names']))
        ]
    return data
