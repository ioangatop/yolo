import random
import yaml


def load_data_cfg(data_cfg, merge_classes=False):
    with open(data_cfg) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)

    if not data.get('colors'):
        data['colors'] = [
            [random.randint(0, 255) for _ in range(3)]
            for _ in range(len(data['names']))
        ]

    if merge_classes:
        data['nc'] = 1
        data['names'] = ['item']

    assert len(data['names']) == data['nc'], f'len(`names`) != `nc` in {data_cfg}.'
    return data
