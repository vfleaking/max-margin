import tensorflow as tf
from models import *
import core

FLAGS = {
    'dataset': 'cifar10',
    'print': True,
    'arch': 'simple_sequential',
    'lr': ('loss-based-v1', 0.1, 2 ** (1 / 5), 2 ** (1 / 10)),
    'archive': 50,
    'E': 10000,
    'B': 100,
    'train_mode': TrainMode.USE_FRAME,
    'model': {
        'description': [
            ['C', 64, [3, 3]],
            ['C', 64, [3, 3]],
            ['P'],
            ['C', 128, [3, 3]],
            ['C', 128, [3, 3]],
            ['P'],
            ['C', 256, [3, 3]],
            ['C', 256, [3, 3]],
            ['C', 256, [3, 3]],
            ['P'],
            ['C', 512, [3, 3]],
            ['C', 512, [3, 3]],
            ['C', 512, [3, 3]],
            ['P'],
            ['C', 512, [3, 3]],
            ['C', 512, [3, 3]],
            ['C', 512, [3, 3]],
            ['P'],
            ['F', 10, {'act': False}]
        ],
        'bias_mode': BiasMode.ALL,
        'init': {
            'weight': 'he_normal',
        },
        'opt': 'gd'
    },
    'loss_ub': 500.0
}

def main(argv):
    core.run(FLAGS)

if __name__ == '__main__':
    core.app_start()
