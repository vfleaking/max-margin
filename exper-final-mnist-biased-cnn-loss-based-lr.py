import tensorflow as tf
from models import *
import core

FLAGS = {
    'dataset': 'mnist',
    'print': True,
    'arch': 'simple_sequential',
    'lr': ('loss-based-v1', 0.1, 2 ** (1 / 5), 2 ** (1 / 10)),
    'archive': ('logloss', 10 ** 5),
    'E': 10000,
    'B': 100,
    'train_mode': TrainMode.USE_FRAME,
    'model': {
        'description': [
            ['C', 32, [5, 5]],
            ['P'],
            ['C', 64, [3, 3]],
            ['P'],
            ['F', 1024],
            ['F', 10, { 'act': False }]
        ],
        'bias_mode': BiasMode.ALL,
        'init': {
            'weight': 'he_normal',
        },
        'opt': 'gd'
    },
    'loss_ub': 600.0
}

def main(argv):
    core.run(FLAGS)

if __name__ == '__main__':
    core.app_start()
