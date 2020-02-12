import importlib
from models import *
import core

exper = importlib.import_module('exper-final-mnist-cnn-loss-based-lr', __package__)

FLAGS = {
    'task': 'attack',
    'eid': [10000],
    'attack_type': core.AttackType.L_2,
    'attack_N': 'all',
    'attack_B': 128,
    'attack_normalize_output': True,
    'attack_initial_const': 10.0,
    **exper.FLAGS,
    'lr': 0.01,
}

def main(argv):
    core.run(FLAGS)

if __name__ == '__main__':
    core.app_start()