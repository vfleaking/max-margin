import tensorflow as tf

from models import TrainMode
from .base_model import *
from .initializers import *

from numpy import linalg as LA
import math

class SimpleSequential(Model):

    def conv_vars(self, name, cin, cout, ks, bias=False):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            cin = int(cin)
            cout = int(cout)
            ks = list(ks)
            f = self.get_variable("f", shape=[ks[0], ks[1], cin, cout], initializer=self.init_policy.weight_init())
            if not bias:
                return f,
            b = self.get_variable("b", shape=[cout], initializer=self.init_policy.bias_init())
            return f, b

    def fc_vars(self, name, cin, cout, bias=False):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            cin = int(cin)
            cout = int(cout)
            A = self.get_variable("W", shape=[cin, cout], initializer=self.init_policy.weight_init())
            if not bias:
                return A,
            b = self.get_variable("b", shape=[cout], initializer=self.init_policy.bias_init())
            return A, b

    def add_conv(self, name, nc, ks, bias=True, act=True):
        vs = self.conv_vars(name, self.y_.shape[-1], nc, ks, bias=bias)
        if self._first_predict:
            self.h_vars.append(vs)
        self.y_ = self.conv(self.y_, *vs)

        if act:
            self.y_ = self.activation(self.y_)

    def add_fc(self, name, cout, bias=True, act=True):
        vs = self.fc_vars(name, self.y_.shape[-1], cout, bias=bias)
        if self._first_predict:
            self.h_vars.append(vs)
        self.y_ = self.fc(self.y_, *vs)

        if act:
            self.y_ = self.activation(self.y_)

    def add_pool(self):
        self.y_ = tf.nn.max_pool(self.y_, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    def add_avg_pool(self):
        self.y_ = tf.nn.avg_pool(self.y_, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

    def predict(self, x):
        self.y_ = x

        layer_id = 0
        has_fc = False
        is_first = True
        for layer_description in self.description:
            layer_id += 1

            if not isinstance(layer_description, (list, tuple)):
                layer_description = [layer_description]

            layer_type = layer_description[0]
            assert isinstance(layer_description[0], str)

            layer_kwargs = {}
            if layer_type in ['C', 'F']:
                if self.bias_mode == BiasMode.NONE:
                    layer_kwargs['bias'] = False
                elif self.bias_mode == BiasMode.ALL:
                    layer_kwargs['bias'] = True
                else:
                    assert self.bias_mode == BiasMode.FIRST_LAYER
                    layer_kwargs['bias'] = is_first
                is_first = False

            if isinstance(layer_description[-1], dict):
                layer_args = layer_description[1:-1]
                layer_kwargs.update(layer_description[-1])
            else:
                layer_args = layer_description[1:]

            if layer_type == 'C':
                self.add_conv('conv' + str(layer_id), *layer_args, **layer_kwargs)
            elif layer_type == 'F':
                if not has_fc:
                    self.y_ = tf.layers.Flatten()(self.y_)
                    has_fc = True
                self.add_fc('fc' + str(layer_id), *layer_args, **layer_kwargs)
            elif layer_type == 'P':
                self.add_pool()
            else:
                assert layer_type == 'A'
                self.add_avg_pool()

        return self.y_

    def __init__(self, name, input_shape, description,
                 bias_mode = BiasMode.FIRST_LAYER, train_mode=TrainMode.USE_FRAME,
                 opt='gd', init={}, loss_type=None, floss=None, act_type='relu'):
        super(SimpleSequential, self).__init__(name)

        self.input_shape = list(input_shape)
        self.init_policy = InitPolicy(init)

        self.description = description
        self.bias_mode = bias_mode
        self.act_type = act_type
        self.train_mode = train_mode

        self.x = tf.placeholder(tf.float32, shape=[None] + self.input_shape, name="placeholder/x")
        self.y = tf.placeholder(tf.int64, shape=[None], name="placeholder/y")
        self.y_ = None
        self.lr = tf.placeholder(tf.float32, name="placeholder/lr")

        self._first_predict = True
        self.h_vars = []
        self.predict(self.x)
        self.nL = len(self.h_vars)
        self.nC = self.y_.shape[-1]

        self._first_predict = False

        self.loss_type = loss_type

        self.build_classification()

        self.build_opt(opt, floss)

    def activation(self, x):
        if self.act_type == 'relu':
            return tf.nn.relu(x)
        elif self.act_type == 'linear':
            return x
        else:
            assert self.act_type == 'relu2'
            return tf.square(tf.nn.relu(x))

    @property
    def is_homogeneous(self):
        return self.bias_mode != BiasMode.ALL or self.nL == 1

    @property
    def image_size(self):
        assert self.input_shape[0] == self.input_shape[1]
        return self.input_shape[0]

    @property
    def num_channels(self):
        return self.input_shape[2]

    @property
    def num_labels(self):
        return self.nC

    def get_stats(self, dataset, log_frame=None):
        if self.is_homogeneous:
            # m_vars: values of the variables
            # theta: vectorized m_vars
            # rho: ||theta||_2
            m_vars = tf.get_default_session().run(self.h_vars)
            theta = np.concatenate([v.reshape(-1) for vs in m_vars for v in vs])
            rho = LA.norm(theta)

            rhos = []

            assert len(m_vars) == self.nL

            if self.act_type != 'relu2':
                norm_prod = 1
                for vs in m_vars:
                    rhoi = LA.norm(np.concatenate([v.reshape(-1) for v in vs]))
                    rhos.append(rhoi)
                    norm_prod *= rhoi
                homo_order = self.nL
            else:
                norm_prod = 1
                for vs in m_vars:
                    rhoi = LA.norm(np.concatenate([v.reshape(-1) for v in vs]))
                    rhos.append(rhoi)
                    norm_prod = norm_prod * norm_prod * rhoi
                homo_order = 2 ** (self.nL - 1) + 1
        else:
            assert self.act_type != 'relu2'
            
            m_vars = tf.get_default_session().run(self.h_vars)

            ws = []

            for i, vs in enumerate(m_vars):
                if i == 0:
                    ws.append(np.concatenate([v.reshape(-1) for v in vs]))
                else:
                    ws.append(vs[0])

            theta = np.concatenate([w.reshape(-1) for w in ws])
            rho = LA.norm(theta)

            rhos = []

            assert len(m_vars) == self.nL

            norm_prod = 1
            for w in ws:
                rhoi = LA.norm(w)
                rhos.append(rhoi)
                norm_prod *= rhoi
            homo_order = self.nL

        # get dataset
        (x, y), (x_test, y_test) = dataset

        # train_res: a dictionary containing loss, acc, y_ (training set)
        train_res = self.test(x, y, log_frame=log_frame)

        # test_res: a dictionary containing loss, acc, y_ (test set)
        if log_frame is None:
            test_res = self.test(x_test, y_test)
        else:
            test_res = self.test(x_test, y_test, log_frame=0.0)

        y_ = train_res['y_']

        N, C = x.shape[0], y_.shape[1]
        L = self.nL

        q = []
        for i in range(N):
            q.append(y_[i, y[i]] - max((y_[i, j] for j in range(C) if j != y[i])))
        q = np.array(q)

        q_min = np.min(q)

        res = {
            'rho': rho,
            'norm_prod': norm_prod,
            'q_min': q_min,
            'q_max': np.max(q),
            'loss': train_res['loss'], 'acc': train_res['acc'],
            'test_loss': test_res['loss'], 'test_acc': test_res['acc']
        }

        extra = {
            'q': q
        }


        for i, rhoi in enumerate(rhos):
            res['rho_%d' % i] = rhoi

        res.update({
            'rhoL': rho ** homo_order,
            'q_bar_min': q_min / rho ** homo_order,
            'q_bar2_min': q_min / norm_prod,
            'n_supp_1': len([q[n] for n in range(N) if q[n] < q_min + math.log(1e1)]),
            'n_supp_2': len([q[n] for n in range(N) if q[n] < q_min + math.log(1e2)]),
            'n_supp_3': len([q[n] for n in range(N) if q[n] < q_min + math.log(1e3)]),
            'n_supp_4': len([q[n] for n in range(N) if q[n] < q_min + math.log(1e4)]),
            'n_supp_5': len([q[n] for n in range(N) if q[n] < q_min + math.log(1e5)]),
            'n_supp_6': len([q[n] for n in range(N) if q[n] < q_min + math.log(1e6)]),
            'n_supp_7': len([q[n] for n in range(N) if q[n] < q_min + math.log(1e7)])
        })

        return res, extra