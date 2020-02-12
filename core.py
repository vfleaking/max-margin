import tensorflow as tf
import pickle
import numpy as np
from collections import Iterable
import pprint
import os, sys
import datetime
import random
import math
import time
import foolbox

from models import *


class AttackType(Enum):
    L_2 = 1
    L_INF = 2
    L_2_BIM = 3
    L_2_DEEPFOOL = 4


def get_dataset(name):
    if name == 'mnist':
        dataset = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = dataset.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0

        x_train, x_test = np.expand_dims(x_train, -1), np.expand_dims(x_test, -1)

        return (x_train, y_train), (x_test, y_test)
    elif name == 'cifar10':
        dataset = tf.keras.datasets.cifar10
        (x_train, y_train),(x_test, y_test) = dataset.load_data()
        x_train, x_test = x_train / 255.0, x_test / 255.0
        y_train, y_test = y_train[:, 0], y_test[:, 0]
        return (x_train, y_train), (x_test, y_test)
    else:
        assert False


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def print_stats(stats: dict) -> None:
    print("{")
    for name in sorted(stats):
        if isinstance(stats[name], int):
            print("    %s: %d" % (name, stats[name]))
        else:
            print("    %s: %.8f    (E: %.8E)" % (name, stats[name], stats[name]))
    print("}")


class Core:
    def __init__(self, flags):
        self.flags = {
            'task': 'train',
            'E': 100,
            'B': 128,
            'archive': 'idle',
            'gpu': 0,
            'arch': 'simple_sequential',
            'train_mode': TrainMode.NORMAL,
            'log_frame0': math.log(math.log(10)),
            'eval_log': None,

            'include_all_rt_loss_in_res': False,

            'lr_lb': 1e-50, # lr lower bound
            'loss_ub': 100.0, # loss upper bound

            # for restoring model
            'eid': 'last',
            'attack_type': AttackType.L_2,
            'attack_set': 'train',
            'attack_normalize_output': False, # for L_2
            'attack_initial_const': 1e-3,
            'attack_B': 64,
            'attack_N': 'all',
            'attack_eps_list': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.3],# for L_INF
            'attack_K': 100, # for L_INF
            'attack_E': 50,  # for L_INF
            'attack': None
        }
        arg_flags = tf.app.flags.FLAGS
        self.flags.update(dict((name, arg_flags[name].value) for name in arg_flags))
        self.flags.update(flags)

        print("{")
        for name in self.flags:
            print("    %s: %s" % (name, self.flags[name]))
        print("}")
        sys.stdout.flush()

        assert 'log' in self.flags

        self.logname = self.flags['log']
        self.resname = self.logname + '/res'
        self.modelname = self.logname + '/model'

        self.evallogname = self.flags['eval_log']
        if self.evallogname is not None:
            self.evalresname = self.evallogname + '/res'

        self.nE = self.flags['E']
        self.nB = self.flags['B']
        self.train_mode = self.flags['train_mode']

        self.nN = None
        self.dataset = None
        self.x_train = self.y_train = self.x_test = self.y_test = None
        self.box_min = self.box_max = None
        self.mod = None

        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.flags['gpu'])

    def new_session(self):
        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        return tf.Session(config=sess_config)

    def fetch_dataset(self) -> None:
        self.dataset = get_dataset(self.flags['dataset'])
        (self.x_train, self.y_train), (self.x_test, self.y_test) = self.dataset
        self.nN = np.shape(self.x_train)[0]
        self.box_min = 0.0
        self.box_max = 1.0

    def build_model(self) -> None:
        assert self.dataset is not None
        assert self.flags['arch'] == 'simple_sequential'
        self.mod = SimpleSequential(self.modelname, self.x_train.shape[1:], train_mode=self.train_mode, **self.flags['model'])

    def get_eids(self):
        eids = self.flags['eid']

        if eids == 'last':
            eids = self.nE
        if isinstance(eids, int):
            eids = [eids]

        if eids == 'all':
            eids = range(self.nS, self.nE + self.nS, self.nS)

        return eids

    def get_lr_sched(self) -> LRScheduler:
        description = self.flags['lr']
        if isinstance(description, float):
            return ConstLR(description)

        assert isinstance(description, tuple)

        return {
            'aggressive': AggressiveLR,
            'aggressive-v2': AggressiveLRv2,
            'aggressive-v3': AggressiveLRv3,
            'loss-based-v1': LossBasedLRv1,
            'const-lr-and-loss-based-lr': ConstLRAndLossBasedLR,
        }[description[0]](*description[1:])

    def get_archiver(self) -> Archiver:
        description = self.flags['archive']
        if isinstance(description, int):
            return ConstEpochArchiver(description)
        if isinstance(description, str):
            description = (description, )

        assert isinstance(description, tuple)

        return {
            'idle': IdleArchiver,
            'logloss': LogLossArchiver,
            'two-phase-const-epoch': TwoPhaseConstEpochArchiver,
        }[description[0]](*description[1:])

    def train_one_batch(self, batch, lr, log_frame=None):
        feed_dict = {
            self.mod.x: self.x_train[batch], self.mod.y: self.y_train[batch],
            self.mod.lr: lr
        }
        if self.train_mode == TrainMode.USE_FRAME:
            feed_dict[self.mod.log_frame] = log_frame

        _, loss = tf.get_default_session().run([self.mod.train_step, self.mod.loss], feed_dict)
        return loss

    def train_one_epoch(self, eid, lr, log_frame=None):
        start_time = time.clock()

        sess = tf.get_default_session()

        status_str = 'lr=%.3e' % lr
        if self.train_mode == TrainMode.USE_FRAME:
            status_str += ' log_frame=%.5f' % log_frame

        all_rt_loss = []

        try:
            perm = np.random.permutation(self.nN)

            for bid in range(self.nN // self.nB):
                batch = perm[bid * self.nB: (bid + 1) * self.nB]
                loss = self.train_one_batch(batch, lr, log_frame)
                all_rt_loss.append(loss)
                if loss > self.flags['loss_ub']:
                    raise Exception('loss is too large: ' + str(loss))

            rt_loss = np.mean(all_rt_loss)

            # don't set print to False if you want to use an adaptive lr scheduler
            if self.flags['print'] or eid == self.nE:
                stats, extra = self.mod.get_stats(self.dataset, log_frame)
            else:
                stats, extra = None, None

            info = {
                'eid': eid,
                'all_rt_loss': all_rt_loss if self.flags['include_all_rt_loss_in_res'] else None,
                'lr': lr,
                'log_frame': log_frame,
                'rt_loss': rt_loss,
                'stats': stats,
                'time': time.clock() - start_time
            }

            if self.flags['print'] or eid == self.nE:
                print('epoch #%d (%s): rt_loss=%.5f loss=%.5f acc=%.3f%% test_loss=%.5f test_acc=%.3f%% [time=%.3fs]'
                      % (eid, status_str, rt_loss, stats['loss'], stats['acc'] * 100, stats['test_loss'], stats['test_acc'] * 100, info['time']))
                print_stats(stats)
                sys.stdout.flush()

            return info, extra

        except Exception as e:
            info = {
                'eid': eid,
                'all_rt_loss': all_rt_loss if self.flags['include_all_rt_loss_in_res'] else None,
                'lr': lr,
                'log_frame': log_frame,
                'stats': None,
                'time': time.clock() - start_time,
            }
            print("epoch #%d (%s): %s [time=%.3fs]" % (eid, status_str, str(e), info['time']))
            return info, None


    def train(self) -> dict:
        self.fetch_dataset()
        self.build_model()
        tf.get_default_graph().finalize()

        lr_sched = self.get_lr_sched()
        archiver = self.get_archiver()

        res = {
            'flags': self.flags
        }

        with self.new_session() as sess:

            sess.run(self.mod.global_init)

            if self.train_mode == TrainMode.USE_FRAME:
                cur_log_frame = self.flags['log_frame0']
                assert isinstance(lr_sched, LRSchedulerWithFrame)
                lr_sched.use_frame(cur_log_frame)
            else:
                cur_log_frame = None

            res['raw'] = {}
            lr_cmd, cur_lr = lr_sched.get()
            assert lr_cmd == 'go'

            self.mod.save(0)

            for eid in range(1, self.nE + 1):

                while True:
                    info, _ = self.train_one_epoch(eid, cur_lr, cur_log_frame)
                    lr_cmd, cur_lr = lr_sched.get(info)
                    if cur_lr < self.flags['lr_lb']:
                        print('next lr is too small: %.3e', cur_lr)
                        exit()
                    if lr_cmd == 'go':
                        break
                    print('rolling back...', end='')
                    sys.stdout.flush()
                    self.mod.restore(eid - 1)
                    print('done')
                    sys.stdout.flush()

                if self.train_mode == TrainMode.USE_FRAME:
                    cur_log_frame += math.log(info['stats']['loss'])

                self.mod.save(eid)
                archiver.run(self.modelname, info)

                res['raw'][eid] = info

                if eid % 10 == 0 or eid == self.nE:
                    with open(self.resname, 'wb') as fp:
                        pickle.dump(res, fp)
                    print(self.resname + ' updated')
                    sys.stdout.flush()

        return res

    def train_peek(self) -> dict:
        with tf.Graph().as_default():
            eids = self.get_eids()

            self.fetch_dataset()
            self.build_model()

            res = {}

            with self.new_session() as sess:
                for eid in eids:
                    self.mod.restore(eid)

                    res[eid] = []

                    res[eid].append(self.mod.get_stats(self.dataset, self.flags['log_frame0']))

                    for i in range(eid, eid + self.nE):

                        if self.train_mode == TrainMode.NORMAL:
                            info, extra = self.train_one_epoch(i, self.flags['lr'])
                        else:
                            info, extra = self.train_one_epoch(i, self.flags['lr'], self.flags['log_frame0'])
                        res[eid].append((info, extra))

            return res

    def attack(self) -> dict:
        assert self.evallogname is not None

        with tf.Graph().as_default():
            eids = self.get_eids()

            self.fetch_dataset()
            self.build_model()

            if self.flags['attack_set'] == 'train':
                x_attack, y_attack = self.x_train, self.y_train
            else:
                x_attack, y_attack = self.x_test, self.y_test

            if self.flags['attack'] is not None:
                attack_perm = self.flags['attack']
                attack_N = len(attack_perm)
            else:
                attack_N = self.flags['attack_N']
                if attack_N == 'all':
                    attack_N = np.shape(x_attack)[0]
                attack_perm = list(range(attack_N))

            B = self.flags['attack_B']

            batch_padding = self.flags['attack_type'] == AttackType.L_2

            if batch_padding:
                attack_perm.extend([random.randint(0, np.shape(x_attack)[0] - 1) for _ in range((B - attack_N % B) % B)])
                assert len(attack_perm) % B == 0

            ae = {}

            def pr(i, x):
                print(i)
                sys.stdout.flush()
                return x

            with self.new_session() as sess:
                if self.flags['attack_type'] == AttackType.L_2:
                    att = CarliniL2(
                        tf.get_default_session(), self.mod, targeted=False, batch_size=B,
                        learning_rate=self.flags['lr'],
                        normalize_output=self.flags['attack_normalize_output'],
                        initial_const=self.flags['attack_initial_const'],
                        boxmin=self.box_min, boxmax=self.box_max
                    )
                    attack_func = lambda x, y: att.attack(x, y)
                    use_one_hot = True
                    need_calc = True
                elif self.flags['attack_type'] == AttackType.L_INF:
                    att = LinfPGDAttack(
                        self.mod, self.flags['attack_eps_list'], self.flags['attack_K'],
                        self.flags['lr'], self.flags['attack_E']
                    )
                    attack_func = lambda x, y: att.attack(x, y)
                    use_one_hot = False
                    need_calc = False
                elif self.flags['attack_type'] == AttackType.L_2_BIM:
                    normalized_y =  self.mod.y_ / tf.reduce_max(tf.abs(self.mod.y_))
                    att_model = foolbox.models.TensorFlowModel(self.mod.x, normalized_y, bounds=(self.box_min, self.box_max))
                    att = foolbox.attacks.L2BasicIterativeAttack(att_model)
                    attack_func = lambda x, y: np.array([att(x[i], pr(i, y[i]), stepsize=0.05, iterations=10) for i in range(np.shape(x)[0])])
                    use_one_hot = False
                    need_calc = True
                elif self.flags['attack_type'] == AttackType.L_2_DEEPFOOL:
                    normalized_y = self.mod.y_ / tf.reduce_max(tf.abs(self.mod.y_))
                    att_model = foolbox.models.TensorFlowModel(self.mod.x, normalized_y, bounds=(0, 1))
                    att = foolbox.attacks.DeepFoolL2Attack(att_model)
                    attack_func = lambda x, y: np.array([att(x[i], pr(i, y[i]), steps=1000) for i in range(np.shape(x)[0])])
                    use_one_hot = False
                    need_calc = True

                tf.get_default_graph().finalize()

                for bid in range((len(attack_perm) + B - 1 ) // B):
                    cur_batch = attack_perm[bid * B: min((bid + 1) * B, len(attack_perm))]

                    x, y = x_attack[cur_batch], y_attack[cur_batch]

                    if use_one_hot:
                        y = one_hot(y, self.mod.num_labels)

                    for eid in eids:
                        self.mod.restore(eid)
                        print('eid: ' + str(eid), flush=True)

                        if not need_calc:
                            cur_ae_images, cur_ae_errors = attack_func(x, y)
                        else:
                            cur_ae_images = attack_func(x, y)
                            cur_ae_errors = np.sum((cur_ae_images - x) ** 2, axis=(1, 2, 3))

                        if batch_padding:
                            realB = min((bid + 1) * B, attack_N) - bid * B
                            cur_ae_images = cur_ae_images[:realB]
                            cur_ae_errors = cur_ae_errors[:realB]

                        if eid not in ae:
                            ae[eid] = {'images': cur_ae_images, 'errors': cur_ae_errors}
                        else:
                            ae[eid]['images'] = np.concatenate([ae[eid]['images'], cur_ae_images])
                            ae[eid]['errors'] = np.concatenate([ae[eid]['errors'], cur_ae_errors])

                    if self.flags['print']:
                        with open(self.evalresname, 'wb') as fp:
                            pickle.dump(ae, fp)
                        print('%s updated: %d images' % (self.evalresname, (bid + 1) * B), flush=True)

            return ae
    
    def run(self) -> dict:
        if self.flags['task'] == 'train':
            return self.train()
        elif self.flags['task'] == 'train_peek':
            return self.train_peek()
        elif self.flags['task'] == 'attack':
            return self.attack()


def app_start() -> None:
    tf.app.flags.DEFINE_string("log", "logs/temp", "log path")
    tf.app.flags.DEFINE_string("eval_log", None, "eval log path")
    tf.app.flags.DEFINE_integer("gpu", 0, "gpu id")
    tf.app.run()

def run(flags: dict) -> dict:
    return Core(flags).run()
