import tensorflow as tf
import pickle
import numpy as np
from collections import Iterable
import pprint
import os
import datetime
import random
from tensorflow.python.training import moving_averages
from enum import Enum
from abc import ABC, abstractmethod


def axes_wo_the_last(x):
    return list(range(len(x.get_shape()) - 1))


class BiasMode(Enum):
    NONE = 1
    FIRST_LAYER = 2
    ALL = 3

class TrainMode(Enum):
    NORMAL = 1
    USE_FRAME = 2

def separate_logits(y, y_, dtype=None):
    C = y_.shape[-1]
    one_hot_y = tf.one_hot(y, y_.shape[-1], on_value=True, off_value=False)
    correct = tf.reshape(tf.boolean_mask(y_, one_hot_y), [-1, 1])
    wrong = tf.reshape(tf.boolean_mask(y_, tf.logical_not(one_hot_y)), [-1, C - 1])

    if dtype is not None:
        correct = tf.cast(correct, tf.float64)
        wrong = tf.cast(wrong, tf.float64)
    return correct, wrong

def hp_sparse_softmax_cross_entropy_with_logits_v1(y, y_):
    # hp stands for high precision

    c, w = separate_logits(y, y_, dtype=tf.float64)
    return tf.math.softplus(tf.reduce_logsumexp(w - c, axis=1))

class Model(ABC):
    def hp_sparse_softmax_cross_entropy_with_logits_v2(self, y, y_):
        # hp stands for high precision

        assert self.log_frame is not None

        c, w = separate_logits(y, y_, dtype=tf.float64)

        return tf.cond(
            tf.reduce_max(w - c) < -30,
            true_fn =lambda: tf.reduce_sum(tf.exp(w - c - self.log_frame), axis=1),
            false_fn=lambda: tf.math.softplus(tf.reduce_logsumexp(w - c, axis=1)) / self.frame
        )


    def hp_sparse_softmax_cross_entropy_with_logits(self, y, y_):
        """
        hp stands for high precision.
        
        The old version of this function:
        y_ = tf.cast(y_, tf.float64)
        one_hot_y = tf.one_hot(y, y_.shape[-1], dtype=tf.float64)
        gap = y_ - tf.expand_dims(tf.reduce_sum(y_ * one_hot_y, -1), -1)
        egap = tf.exp(gap) - one_hot_y
        s = tf.reduce_sum(egap, -1)
        all_loss = tf.log1p(s)
        def grad(dy):
            se = tf.expand_dims(s, -1)
            inv = tf.expand_dims(dy / (1 + s), -1)
            return tf.zeros(shape=tf.shape(y)), tf.cast((egap - se * one_hot_y) * inv, tf.float32)
        return all_loss, grad
        """

        if self.log_frame is None:
            return hp_sparse_softmax_cross_entropy_with_logits_v1(y, y_)
        else:
            return self.hp_sparse_softmax_cross_entropy_with_logits_v2(y, y_)

    def fc(self, x, A, b=None):
        x = tf.matmul(x, A)
        if b is not None:
            x = x + b
        return x

    def conv(self, x, f, b=None):
        x = tf.nn.conv2d(x, f, [1, 1, 1, 1], 'SAME')
        if b is not None:
            x = x + b
        return x

    def get_variable(self, *args, **kwargs):
        v = tf.get_variable(*args, **kwargs)
        return tf.debugging.check_numerics(v, v.name)

    def __init__(self, name):
        self.name = name
        self.vars = None
        self.all_loss = None
        self.all_acc = None
        self.reg_loss = None
        self.loss = None
        self.acc = None
        self.global_init = None
        self.grads = None
        self.train_step = None
        self.lr = None
        self.saver = None

        self.train_mode = TrainMode.NORMAL
        self.loss_type = None # default: softmax crossentropy

        # If train_mode = USE_FRAME:
        #     log_frame is not None
        #     frame is expected to be a good approximation for loss
        #     real loss = loss * exp(log_frame)
        #     real grad = grad * exp(log_frame)
        #     real lr   = lr   * exp(-log_frame)
        self.log_frame = None
        self.frame = None

    def build_classification(self):
        self.vars = tf.trainable_variables()
        self.saver = tf.train.Saver(self.vars, max_to_keep=1)

        if self.train_mode == TrainMode.USE_FRAME:
            self.log_frame = tf.placeholder(tf.float64)
            self.frame = tf.exp(self.log_frame) # warning: if log_frame -> inf then frame -> 0

        if self.loss_type is None:
            hp_all_loss = self.hp_sparse_softmax_cross_entropy_with_logits(self.y, self.y_)
        else:
            assert False

        self.all_loss = tf.debugging.check_numerics(hp_all_loss, 'all_loss')

        self.loss = tf.reduce_mean(self.all_loss)
        self.loss = tf.debugging.check_numerics(self.loss, 'loss')

        # self.all_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        #        labels=self.y,
        #        logits=self.y_)
        # self.loss = tf.reduce_mean(hp_all_loss)

        if hasattr(self, 'weight_decay'):
            self.reg_loss = self.weight_decay * sum((tf.nn.l2_loss(v) for v in self.vars))
            self.loss = self.loss + tf.cast(self.reg_loss, tf.float64)
        
        self.all_acc = tf.cast(tf.equal(tf.argmax(self.y_, axis=1), self.y),
                               tf.float32)
        self.acc = tf.reduce_mean(self.all_acc)

    def build_opt(self, opt, floss=None):
        self.global_init = tf.global_variables_initializer()

        if opt == 'gd':
            self.lr = tf.placeholder(tf.float64)
            if self.train_mode == TrainMode.NORMAL:
                if floss is None:
                    self.grads = tf.gradients(self.lr * self.loss, self.vars)
                else:
                    self.grads = tf.gradients(self.lr * floss(self.loss), self.vars)
            else:
                assert self.train_mode == TrainMode.USE_FRAME
                assert floss is None
                self.grads = tf.gradients(self.lr * self.loss, self.vars)

            for i in range(len(self.vars)):
                self.grads[i] = tf.debugging.check_numerics(self.grads[i], 'grad of ' + self.vars[i].name)

            ops = []
            for v, g in zip(self.vars, self.grads):
                ops.append(tf.assign_sub(v, g))
            self.train_step = tf.group(*ops)
        else:
            assert False

    def test(self, x, y, B=128, log_frame=None):
        sess = tf.get_default_session()

        N = x.shape[0]

        L = (N + B - 1) // B

        all_loss = []
        all_acc = []
        all_y_ = []
        for b in range(L):
            bl = b * B
            br = min((b + 1) * B, N)

            feed_dict = {
                self.x: x[bl:br], self.y: y[bl:br]
            }
            if log_frame is not None:
                feed_dict[self.log_frame] = log_frame

            cur_all_loss, cur_all_acc, cur_y_ = sess.run([self.all_loss, self.all_acc, self.y_], feed_dict)
            all_loss += list(cur_all_loss)
            all_acc += list(cur_all_acc)
            all_y_.append(cur_y_)

        all_y_ = np.concatenate(all_y_)

        assert len(all_loss) == N and len(all_acc) == N
        assert all_y_.shape[0] == N

        return {
            'loss': np.mean(all_loss),
            'acc': np.mean(all_acc),
            'y_': all_y_
        }

    @abstractmethod
    def get_stats(self, dataset):
        pass

    def save(self, global_step):
        self.saver.save(tf.get_default_session(), self.name, global_step)

    def restore(self, global_step):
        self.saver.restore(tf.get_default_session(), self.name + '-' + str(global_step))
