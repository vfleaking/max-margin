import importlib
from models import *
import core

exper = importlib.import_module('exper-final-mnist-cnn-loss-based-lr', __package__)

FLAGS = {
    'task': 'attack',
    'eid': [10000],
    'attack_type': core.AttackType.L_2_BIM,
    'attack_N': 'all',
    'attack_B': 128,
    **exper.FLAGS
}

def main(argv):
    core.run(FLAGS)

if __name__ == '__main__':
    core.app_start()