import tensorflow as tf

from .base_model import *

def rescaling_initializer(lam, initializer):
    def init(shape, dtype=None, partition_info=None):
        return initializer(shape, dtype, partition_info) * lam
    return init

def normalized_initializer(norm, initializer):
    def init(shape, dtype=None, partition_info=None):
        v = initializer(shape, dtype, partition_info)
        return v * (norm / tf.sqrt(tf.reduce_sum(tf.square(v))))
    return init

class InitPolicy:
    def __init__(self, config={}):
        self.config = {
            'weight': 'glorot_uniform',
            'weight_normalized': False,
            'weight_rescale': 1.0,
            'bias': 'zeros',
            'bias_normalized': False,
            'bias_rescale': 1.0,
        }
        self.config.update(config)

        for name in ['weight', 'bias']:
            if isinstance(self.config[name], str):
                self.config[name] = {
                    'glorot_uniform': tf.initializers.glorot_uniform(),
                    'glorot_normal': tf.initializers.glorot_normal(),
                    'he_normal': tf.initializers.he_normal(),
                    'he_uniform': tf.initializers.he_uniform(),
                    'truncated_normal': tf.initializers.truncated_normal(),
                    'random_normal': tf.initializers.random_normal(),
                    'zeros': tf.initializers.zeros(),
                }[self.config[name]]

    def weight_init(self):
        init = self.config['weight']
        if self.config['weight_normalized']:
            init = normalized_initializer(self.config['weight_rescale'], init)
        elif self.config['weight_rescale'] != 1.0:
            init = rescaling_initializer(self.config['weight_rescale'], init)
        return init

    def bias_init(self):
        init = self.config['bias']
        if self.config['bias_normalized']:
            init = normalized_initializer(self.config['bias_rescale'], init)
        elif self.config['bias_rescale'] != 1.0:
            init = rescaling_initializer(self.config['bias_rescale'], init)
        return init