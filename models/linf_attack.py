"""
Modified from https://github.com/MadryLab/adversarial_spatial/blob/master/pgd_attack.py
"""
import tensorflow as tf
import numpy as np


class LinfPGDAttack:
    def __init__(self, model, eps_list, k, a, repeat, random_start=True, loss_func='cw'):
        """Attack parameter initialization. The attack performs k steps of
           size a, while always staying within epsilon from the initial
           point."""
        self.model = model
        self.eps_list = eps_list
        self.k = k
        self.a = a
        self.repeat = repeat
        self.rand = random_start

        if loss_func == 'xent':
            loss = model.xent
        elif loss_func == 'cw':
            label_mask = tf.one_hot(model.y,
                                    10,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            correct_logit = tf.reduce_sum(label_mask * model.y_, axis=1)
            wrong_logit = tf.reduce_max((1 - label_mask) * model.y_, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.grad = tf.gradients(loss, model.x)[0]

    def attack(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""

        sess = tf.get_default_session()

        nN = x_nat.shape[0]
        active = range(nN)

        ae = {'images': np.zeros(x_nat.shape), 'errors': np.ones(nN)}

        for eps in sorted(self.eps_list):
            for r in range(self.repeat):
                if not active:
                    continue
                x_nat_a = x_nat[active]
                y_a = y[active]

                if self.rand:
                    x = x_nat_a + np.random.uniform(-eps, eps, x_nat_a.shape)
                else:
                    x = np.copy(x_nat_a)

                for i in range(self.k):
                    grad = sess.run(self.grad, feed_dict={self.model.x: x, self.model.y: y_a})

                    x += eps * self.a * np.sign(grad)

                    x = np.clip(x, x_nat_a - eps, x_nat_a + eps)
                    x = np.clip(x, 0, 1)  # ensure valid pixel range

                all_acc = sess.run(self.model.all_acc, feed_dict={self.model.x: x, self.model.y: y_a})

                new_active = []
                for i in range(len(active)):
                    if all_acc[i] == 0.0:
                        ae['images'][active[i]] = x[i]
                        ae['errors'][active[i]] = eps
                    else:
                        new_active.append(active[i])
                active = new_active

                print('eps: ' + str(eps) + ' repeat: ' + str(r) + ' active: ' + str(len(active)), flush=True)

        return ae['images'], ae['errors']
