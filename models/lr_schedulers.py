import tensorflow as tf
import numpy as np
import math

from .base_model import *


class LRScheduler:
    @abstractmethod
    def get(self, info=None):
        pass

class LRSchedulerWithFrame(LRScheduler):
    @abstractmethod
    def use_frame(self, log_frame_0):
        pass


class ConstLR(LRSchedulerWithFrame):

    def __init__(self, lr):
        self.lr = lr
        self.frame_enabled = False
        self.log_frame_0 = None
        self.eid = 0

    def use_frame(self, log_frame_0):
        self.frame_enabled = True
        self.log_frame_0 = log_frame_0

    def get(self, info=None):
        assert info is None or info['stats'] is not None

        if self.frame_enabled:
            if info is None:
                return 'go', self.lr * math.exp(self.log_frame_0)
            else:
                return 'go', self.lr * math.exp(info['log_frame']) * info['stats']['loss']
        else:
            return 'go', self.lr


class AggressiveLR(LRScheduler):

    def __init__(self, lr, cd=5):
        self.lr = lr
        self.eid = 0
        self.last_loss = np.inf
        self.cd = cd
        self.rest_cd = cd

    def get(self, info=None):
        if self.eid == 0:
            self.eid += 1
            return 'go', self.lr

        # check if the last epoch has NaN/inf
        if info['stats'] is None or info['stats']['loss'] > self.last_loss:
            self.lr /= 2.0
            self.rest_cd = self.cd
            return 'back', self.lr
        else:
            self.eid += 1
            self.last_loss = info['stats']['loss']
            self.rest_cd -= 1
            if self.rest_cd == 0:
                self.lr *= 2
                self.rest_cd = self.cd
            return 'go', self.lr


class AggressiveLRv2(LRScheduler):

    def __init__(self, lr, cd=5):
        self.lr = lr
        self.eid = 0
        self.last_loss = np.inf
        self.cd = cd

    def get(self, info=None):
        if self.eid == 0:
            self.eid += 1
            return 'go', self.lr

        # check if the last epoch has NaN/inf
        if info['stats'] is None or info['stats']['loss'] > self.last_loss:
            self.lr /= 2.0
            return 'back', self.lr
        else:
            self.eid += 1
            self.last_loss = info['stats']['loss']
            self.lr *= 2 ** (1 / self.cd)
            return 'go', self.lr


class AggressiveLRv3(LRScheduler):

    def __init__(self, lr, alpha1=2 ** (1 / 5), alpha2=2 ** (1 / 5)):
        self.lr = lr
        self.eid = 0
        self.last_loss = np.inf
        self.alpha1 = alpha1
        self.alpha2 = alpha2

    def get(self, info=None):
        if self.eid == 0:
            self.eid += 1
            return 'go', self.lr

        # check if the last epoch has NaN/inf
        if info['stats'] is None or info['stats']['loss'] > self.last_loss:
            self.lr /= self.alpha2
            return 'back', self.lr
        else:
            self.eid += 1
            self.last_loss = info['stats']['loss']
            self.lr *= self.alpha1
            return 'go', self.lr

class LossBasedLRv1(LRSchedulerWithFrame):

    def __init__(self, lr=0.1, alpha1=2 ** (1 / 5), alpha2=2 ** (1 / 10), eps=0, loss0=math.log(10)):
        self.lr = lr
        self.eid = 0
        self.loss0 = loss0
        self.last_loss = loss0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.frame_enabled = False
        self.eps = eps

    def use_frame(self, log_frame_0):
        self.frame_enabled = True

    def _get(self):
        if not self.frame_enabled:
            return self.lr * self.loss0 / self.last_loss
        else:
            return self.lr

    def get(self, info=None):
        if self.eid == 0:
            self.eid += 1
            return 'go', self._get()

        def dec():
            if info['stats'] is None:
                return False
            if self.frame_enabled:
                return info['stats']['loss'] <= 1 + self.eps
            else:
                return info['stats']['loss'] / self.last_loss <= 1 + self.eps

        # check if the last epoch has NaN/inf
        if not dec():
            self.lr /= self.alpha2
            return 'back', self._get() # re-train the last epoch with a smaller lr
        else:
            self.eid += 1
            self.last_loss = info['stats']['loss']
            self.lr *= self.alpha1
            return 'go', self._get() # train the next epoch with a larger lr

class ConstLRAndLossBasedLR(LRSchedulerWithFrame):
    def __init__(self, lr, alpha1=2 ** (1 / 5), alpha2=2 ** (1 / 5), eps=0, loss0=math.log(10)):
        self.lr = lr
        self.eid = 0
        self.loss0 = loss0
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.frame_enabled = False
        self.log_frame_0 = None
        self.eps = eps

        self.fitted = False

    def use_frame(self, log_frame_0):
        self.frame_enabled = True
        self.log_frame_0 = log_frame_0

    def get(self, info=None):
        assert self.frame_enabled

        if self.eid == 0:
            self.eid += 1
            return 'go', self.lr * math.exp(self.log_frame_0)

        if not self.fitted and info['stats'] is not None and info['stats']['acc'] == 1.0:
            self.fitted = True
            self.lr *= math.exp(info['log_frame']) * info['stats']['loss']

        if not self.fitted:
            return 'go', self.lr * math.exp(info['log_frame']) * info['stats']['loss']

        def dec():
            # check if the last epoch has NaN/inf
            if info['stats'] is None:
                return False
            return info['stats']['loss'] <= 1 + self.eps

        if not dec():
            self.lr /= self.alpha2
            return 'back', self.lr
        else:
            self.eid += 1
            self.lr *= self.alpha1
            return 'go', self.lr