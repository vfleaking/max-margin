import importlib
from models import *
import core

exper = importlib.import_module('exper-final-mnist-cnn-loss-based-lr', __package__)

FLAGS = {
    'task': 'attack',
    'eid': [10000],
    'attack_type': core.AttackType.L_INF,
    'attack_set': 'test',
    'attack_N': 'all',
    'attack_B': 128,
    'attack_E': 10,
    'attack_K': 100,
    # 'attack_eps_list': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3],# for L_INF
    'attack_eps_list': [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19],# for L_INF
    **exper.FLAGS,
    'lr': 0.1
}

def main(argv):
    core.run(FLAGS)

if __name__ == '__main__':
    core.app_start()