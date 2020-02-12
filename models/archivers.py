import tensorflow as tf
import numpy as np
import math
import os

from .base_model import *

def archive(name, eid):
    dirname = os.path.dirname(name)
    basename = os.path.basename(name)
    os.system("cd %s && tar -czf %s-%d.tar.gz %s-%d.*" % (dirname, basename, eid, basename, eid))
    print("%s-%d archived" % (name, eid))


class Archiver:
    @abstractmethod
    def run(self, name, info):
        pass


class IdleArchiver(Archiver):
    def run(self, name, info):
        pass


class ConstEpochArchiver(Archiver):

    def __init__(self, nS):
        self.nS = nS

    def run(self, name, info):
        eid = info['eid']
        if self.nS != 0 and eid % self.nS == 0:
            archive(name, eid)

class TwoPhaseConstEpochArchiver(Archiver):

    def __init__(self, nS1, nS2):
        self.nS1 = nS1
        self.nS2 = nS2

    def run(self, name, info):
        eid = info['eid']
        if (eid < self.nS2 and eid % self.nS1 == 0) or eid % self.nS2 == 0:
            archive(name, eid)


class LogLossArchiver(Archiver):

    def __init__(self, base=10):
        self.base = base
        self.next = math.inf

    def run(self, name, info):
        eid = info['eid']
        if eid == 0:
            archive(name, eid)
        else:
            if info['log_frame'] is None:
                loss = info['stats']['loss']
                if loss < self.next:
                    archive(name, eid)
                    self.next = self.base ** (math.ceil(math.log(loss) / math.log(self.base)) - 1)
            else:
                log_loss = (math.log(info['stats']['loss']) + info['log_frame']) / math.log(self.base)
                if log_loss < self.next:
                    archive(name, eid)
                    self.next = math.ceil(log_loss) - 1