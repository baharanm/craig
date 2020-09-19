import glob
import argparse

import numpy as np
import time
from os import path
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
np.seterr(all='ignore')

import util
import random


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def softmax(x):
    e = np.exp(x - np.max(x))  # prevent overflow
    if e.ndim == 1:
        return e / np.sum(e, axis=0)
    else:
        return e / np.array([np.sum(e, axis=1)]).T  # ndim = 2


class LogisticRegression(object):
    def __init__(self, dim, num_class):
        self.binary = num_class == 1
        self.W = np.zeros((dim, num_class))  # initialize W 0
        self.b = np.zeros(num_class)  # initialize bias 0
        self.params = np.array([self.W, self.b])

    def activation(self, input, params=None):
        W, b = params if params is not None else self.params
        if self.binary:
            return sigmoid(np.dot(input, W) + b)
        else:
            return softmax(np.dot(input, W) + b)

    # regularized_negative_log_likelihood
    def loss(self, input, label, l2_reg=0.00, params=None):
        sigmoid_activation = self.activation(input, params)
        cross_entropy = - np.mean(np.sum(label * np.log(sigmoid_activation) +
                                         (1 - label) * np.log(1 - sigmoid_activation), axis=1))

        return cross_entropy + l2_reg * np.linalg.norm(self.W) ** 2 / 2

    def predict(self, input, params=None):
        return self.activation(input, params)

    def accuracy(self, input, label, params=None):
        if self.binary:
            # Note: label is not one hot encoded
            return np.mean(np.isclose(np.rint(self.predict(input, params)), label))
        else:
            return np.mean(np.argmax(self.predict(input, params), axis=1) == np.argmax(label, axis=1))

    def gradient(self, input, label, l2_reg=0.00, params=None):
        p_y_given_x = self.activation(input, params)
        d_y = label - p_y_given_x
        d_W = -np.dot(np.reshape(input, (1, -1)).T, np.reshape(d_y.T, (1, -1))) - l2_reg * self.W
        d_b = -np.mean(d_y, axis=0)
        return np.array([d_W, d_b])


class Optimizer(object):

    @staticmethod
    def order_elements(shuffle, n, seed=1234):
        if shuffle == 0:
            indices = np.arange(n)
        elif shuffle == 1:
            indices = np.random.permutation(n)
        elif shuffle == 2:
            indices = np.random.randint(0, n, n)
        else:  # fixed permutation
            np.random.seed(seed)
            indices = np.random.permutation(n)
        return indices

    def optimize(self, method, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        if method == 'sgd':
            return self.sgd(model, data, labels, weights, num_epochs, shuffle, lr, l2_reg)
        elif method == 'saga':
            return self.saga(model, data, labels, weights, num_epochs, shuffle, lr, l2_reg)
        elif method == 'svrg':
            return self.svrg(model, data, labels, weights, num_epochs, shuffle, lr, l2_reg)
        else:
            print('Optimizer is not defined!')

    def sgd(self, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        n = len(data)
        W = [[]] * num_epochs
        T = np.empty(num_epochs)

        time.sleep(.1)
        start_epoch = time.process_time()

        for epoch in range(num_epochs):
            indices = self.order_elements(shuffle, n)
            for i in indices:
                grads = model.gradient(data[i], labels[i], l2_reg / n) * weights[i]
                model.params -= lr[epoch] * grads
            W[epoch] = model.params.copy()
            T[epoch] = (time.process_time() - start_epoch)
        return W, T

    def saga(self, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        n = len(data)
        W = [[]] * num_epochs
        T = np.empty(num_epochs)

        time.sleep(.1)
        start_epoch = time.process_time()

        saved_grads = np.array([model.gradient(data[i], labels[i], l2_reg / n) * weights[i] for i in range(n)])
        avg_saved_grads = saved_grads.mean(axis=0)

        for epoch in range(num_epochs):
            indices = self.order_elements(shuffle, n)
            for i in indices:
                grads = model.gradient(data[i], labels[i], l2_reg / n) * weights[i]
                model.params -= lr[epoch] * (grads - saved_grads[i] + avg_saved_grads)
                avg_saved_grads += (grads - saved_grads[i]) / n
                saved_grads[i] = grads

            W[epoch] = model.params.copy()
            T[epoch] = (time.process_time() - start_epoch)
        return W, T

    def svrg(self, model, data, labels, weights, num_epochs, shuffle, lr, l2_reg):
        n = len(data)
        W = [[]] * num_epochs
        T = np.empty(num_epochs)

        time.sleep(.1)
        start_epoch = time.process_time()

        for epoch in range(num_epochs):
            init_grads = np.array([model.gradient(data[i], labels[i], l2_reg / n) * weights[i] for i in range(n)])
            avg_init_grads = np.mean(init_grads, axis=0)

            indices = self.order_elements(shuffle, n)
            for i in indices:
                grads = model.gradient(data[i], labels[i], l2_reg / n) * weights[i]
                model.params -= lr[epoch] * (grads - init_grads[i] + avg_init_grads)

            W[epoch] = model.params.copy()
            T[epoch] = (time.process_time() - start_epoch)
        return W, T


def load_dataset(dataset, normalize=False): # TODO <===
    DATASET_DIR = '/dfs/scratch2/turing1-backup/lfs/turing1/0/baharanm/data/'
    if dataset == 'covtype':
        print(f'Loading {dataset}')
        X, y = util.load_dataset('covtype', DATASET_DIR)
        N = len(X)
        NUM_TRAINING, NUM_VALIDATION = int(N / 2), int(N / 2) + int(N / 4)
        # NUM_TRAINING, NUM_VALIDATION = int(N / 256), int(N / 256) + int(N / 512)
        sample = np.arange(N)
        np.random.seed(0)
        np.random.shuffle(sample)
        train_sample, val_sample, test_sample = \
            sample[:NUM_TRAINING], sample[NUM_TRAINING:NUM_VALIDATION], sample[NUM_VALIDATION:]

        X_train, y_train = X[train_sample, :], y[train_sample]
        X_val, y_val = X[val_sample, :], y[val_sample]
        X_test, y_test = X[test_sample, :], y[test_sample]

    elif dataset == 'ijcnn1':
        print(f'Loading {dataset}')
        X_train, y_train = util.load_dataset('ijcnn1.tr', DATASET_DIR)
        X_test, y_test = util.load_dataset('ijcnn1.t', DATASET_DIR)
        # X_train, y_train = X_train[:500], y_train[:500]
        X_val, y_val = X_test, y_test  # TODO <======================

    elif dataset == 'combined':
        print(f'Loading {dataset}')
        X_train, y_train = util.load_dataset('combined_scale', DATASET_DIR)
        X_test, y_test = util.load_dataset('combined_scale.t', DATASET_DIR)
        # X_train, y_train = X_train[1:200], y_train[1:200]
        X_0, y_0 = X_train[y_train == 0], y_train[y_train == 0]
        X_1, y_1 = X_train[y_train == 1], y_train[y_train == 1]
        X_2, y_2 = X_train[y_train == 2], y_train[y_train == 2]

        X_1, y_1 = X_1[:18266], y_1[:18266]
        X_2, y_2 = X_2[:18266 * 2], y_2[:18266 * 2]
        X_train, y_train = np.vstack([X_0, X_1, X_2]), np.hstack([y_0, y_1, y_2])

        data_mean = np.vstack([X_train, X_test]).mean(axis=0)
        X_train -= data_mean
        X_test -= data_mean
        X_val, y_val = X_test, y_test  # TODO <========================

    if dataset in ['covtype', 'ijcnn1']:
        y_train = np.reshape(y_train, (-1, 1))
        y_val = np.reshape(y_val, (-1, 1))
        y_test = np.reshape(y_test, (-1, 1))
    elif dataset == 'combined':
        num_class = 3
        y_train = np.eye(num_class)[y_train]
        y_val = np.eye(num_class)[y_val]
        y_test = np.eye(num_class)[y_test]

    print(f'Training size: {len(y_train)}, Test size: {len(y_test)}')
    return X_train, y_train, X_val, y_val, X_test, y_test


def get_param_range(subset_size, exp_decay, method, data):
    g_range, b_range = [0], [0]
    if exp_decay > 0 and data == 'ijcnn1':
        if method == 'sgd':
            if subset_size in [0.1, 0.2]:
                g_range = np.arange(10, 30) * .001
                b_range = np.arange(70, 110) * .01
            elif subset_size < 1.0:
                g_range = np.arange(20, 40) * .001
                b_range = np.arange(70, 130) * .01
            else:
                g_range = np.arange(30, 40) * .001
                b_range = np.arange(95, 105) * .01

        elif method == 'saga':
            b_range = [1]
            g_range = np.arange(40, 120, 1) * .0001
        elif method == 'svrg':
            b_range = [1]
            g_range = np.arange(30, 170) * .0001  # for 10% random

    elif exp_decay > 0 and data == 'combined':

        if method == 'sgd':
            g_range = np.arange(10, 50) * .001
            b_range = np.arange(40, 110) * .01

        elif method == 'saga':
            b_range = [1]
            g_range = np.arange(40, 120, 1) * .0001

        elif method == 'svrg':
            b_range = [1]
            g_range = np.arange(50, 120) * .0001

    elif exp_decay > 0 and data == 'covtype':
        if subset_size == .1:
            g_range = np.arange(10, 34) * .001
            b_range = np.arange(84, 96) * .01
        if subset_size == .2:
            g_range = np.arange(16, 40) * .001
            b_range = np.arange(76, 92) * .01
        if subset_size == .3:
            g_range = np.arange(20, 52) * .001
            b_range = np.arange(75, 84) * .01
        if subset_size == .4:
            g_range = np.arange(25, 48) * .001
            b_range = np.arange(71, 82) * .01
        if subset_size == .5:
            g_range = np.arange(28, 50) * .001
            b_range = np.arange(67, 76) * .01
        if subset_size == .6:
            g_range = np.arange(30, 50) * .001
            b_range = np.arange(67, 75) * .01
        if subset_size == .7:
            g_range = np.arange(30, 48) * .001
            b_range = np.arange(65, 73) * .01
        if subset_size == .8:
            g_range = np.arange(33, 43) * .001
            b_range = np.arange(63, 68) * .01
        if subset_size == .9:
            g_range = np.arange(39, 44) * .001
            b_range = np.arange(59, 66) * .01
        if subset_size == 1:
            g_range = np.arange(40, 52) * .001
            b_range = np.arange(50, 55) * .01
    else:
        g_range = [0.1, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.25, 0.3, 0.35]  # already done: 0.01, 0.03, 0.05
        b_range = [0.7, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.82, 0.83, 0.84, 0.85, 0.9, 0.95]
        if subset_size < 1:
            g_range = [0.000035, 0.009, 0.01, 0.013, 0.015, 0.017, 0.018, 0.019, 0.02, 0.025, 0.03]
            b_range = np.arange(0, 19) * .01  # [1, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7]

    # fixed step size for svrg, saga
    if data == 'covtype' and method in ['svrg', 'saga']:
        # NOTE: exp_decay = 1, b = 1 or exp_decay = 0, b = 0
        exp_decay = 1
        b_range = [1]
        if method == 'saga':
            g_range = np.arange(10, 80, 1) * .0001
        elif method == 'svrg':
            g_range = np.arange(15, 120) * .0001
    return g_range, b_range


def test(method='sgd', data='covtype', exp_decay=1, subset_size=1., greedy=1, shuffle=0, g_cnt=-1.,
         b_cnt=-1., num_runs=10, metric='', reg=1e-5, rand='', num_epochs=-1, from_all=0):
    if num_epochs == -1:
        num_epochs = 20 + int(np.ceil((1. / subset_size) * 5)) + 5 if subset_size < 1 else 20  # Todo <-- (+5)
    else:
        rand += f'_e{num_epochs}'
    # num_parts = 1

    train_data, train_target, val_data, val_target, test_data, test_target = load_dataset(data)
    num_class = 1 if data in ['covtype', 'ijcnn1'] else 3

    if g_cnt != -1 and b_cnt != -1:
        g_range = [g_cnt]
        b_range = [b_cnt]
        print(f'Running with b: {b_cnt}, g: {g_cnt}')
    else:
        g_range, b_range = get_param_range(subset_size, exp_decay, method, data)

    folder = f'/tmp/{data}'
    x_runs_f = [[]] * num_runs
    f_runs_f = np.zeros((num_runs, num_epochs))
    ft_runs_f = np.zeros((num_runs, num_epochs))
    acc_runs_f = np.zeros((num_runs, num_epochs))
    t_runs_f = np.zeros((num_runs, num_epochs))
    x_runs_a = [[]] * num_runs
    f_runs_a = np.zeros((num_runs, num_epochs))
    ft_runs_a = np.zeros((num_runs, num_epochs))
    acc_runs_a = np.zeros((num_runs, num_epochs))
    t_runs_a = np.zeros((num_runs, num_epochs))

    for itr in range(num_runs):
        f_best, acc_best, b_f, g_f, b_a, g_a = 1e10, 0, 0, 0, 0, 0

        if greedy == 1:
            file_name = ''
            if from_all == 0 and path.exists(f'{folder}_{subset_size}_{metric}.npz'):
                file_name = glob.glob(f'{folder}_{subset_size}_{metric}.npz')[0]
            elif from_all > 0 and path.exists(f'{folder}_all_{subset_size}_{metric}.npz'):
                file_name = glob.glob(f'{folder}_{subset_size}_{metric}.npz')[0]
            if file_name != '':
                print(f'reading from {file_name}')
                dataset = np.load(f'{file_name}')
                order, weights, total_ordering_time = dataset['order'], dataset['weight'], dataset['order_time']
            else:
                print(f'Calculating the ordering and weights for metric {metric}')
                train_y = np.argmax(train_target, axis=1) if data == 'combined' else np.reshape(train_target, -1)
                if from_all > 0:
                    train_y = np.zeros(np.shape(train_y), dtype=int)
                    folder += '_all'

                util.save_all_orders_and_weights(folder, train_data, metric=metric,
                                                 stoch_greedy=False, y=train_y, equal_num=False)
                return
        else:
            print('Selecting a random subset')
            order = np.arange(0, len(train_data))
            random.shuffle(order)
            order = order[:int(subset_size * len(train_data))]
            weights = np.ones(len(train_data), dtype=np.float)

        print(f'--------------- run number: {itr}, rand: {rand}, '
              f'subset: {subset_size}, subset size: {len(order)}, num_epochs: {num_epochs} -----------------')
        for gamma in g_range:
            for b in b_range:
                dim = len(train_data[0])
                model = LogisticRegression(dim, num_class)
                lr = gamma * np.power(b, np.arange(num_epochs)) if exp_decay else gamma / (1 + b * np.arange(num_epochs))
                x_s, t_s = Optimizer().optimize(
                    method, model, train_data[order, :], train_target[order], weights, num_epochs, shuffle, lr, reg)
                f_s = model.loss(train_data, train_target, l2_reg=reg)
                acc_s = model.accuracy(val_data, val_target)

                print(f'data: {data}, method: {method}, run: {itr}, exp_decay: {exp_decay}, size: {subset_size} {rand} '
                      f'--> f: {f_s}, acc: {acc_s}, b: {b}, g: {gamma}')
                if f_s < f_best:
                    f_best, x_f, g_f, b_f, t_f = f_s, x_s, gamma, b, t_s
                    x_runs_f[itr] = x_f
                    t_runs_f[itr, :] = t_f
                    f_runs_f[itr, :] = [model.loss(train_data, train_target, reg, x_f[j]) for j in range(num_epochs)]
                    ft_runs_f[itr, :] = [model.loss(test_data, test_target, reg, x_f[j]) for j in range(num_epochs)]
                    acc_runs_f[itr, :] = [model.accuracy(test_data, test_target, x_f[j]) for j in range(num_epochs)]
                    print(f'Saving the results to {folder}_{method}_{subset_size}_{rand}_best_f_{metric}_w')
                    np.savez(f'{folder}_{method}_{subset_size}_{rand}_best_f_{metric}_w', g=g_f, b=b_f,
                             X_all=x_runs_f, F_all=f_runs_f, T_all=t_runs_f, Acc_all=acc_runs_f, FT_all=ft_runs_f)
                if acc_s > acc_best:
                    acc_best, x_a, g_a, b_a, t_a = acc_s, x_s, gamma, b, t_s
                    x_runs_a[itr] = x_a
                    t_runs_a[itr, :] = t_a
                    f_runs_a[itr, :] = [model.loss(train_data, train_target, reg, x_a[j]) for j in range(num_epochs)]
                    ft_runs_a[itr, :] = [model.loss(test_data, test_target, reg, x_a[j]) for j in range(num_epochs)]
                    acc_runs_a[itr, :] = [model.accuracy(test_data, test_target, x_a[j]) for j in range(num_epochs)]
                    print(f'Saving the results to {folder}_{method}_{subset_size}_{rand}_best_acc_{metric}_w')
                    np.savez(f'{folder}_{method}_{subset_size}_{rand}_best_acc_{metric}_w', g=g_a, b=b_a,
                             X_all=x_runs_a, F_all=f_runs_a, T_all=t_runs_a, Acc_all=acc_runs_a, FT_all=ft_runs_a)
                print(f'Best solution is => f: {f_best}, a: {acc_best}, b_f: {b_f}, g_f: {g_f}, b_a: {b_a}, g_a: {g_a}')

        print(f'Saving the final results to {folder}_{method}_{subset_size}_{rand}_best_f_{metric}_w')
        np.savez(f'{folder}_{method}_{subset_size}_{rand}_best_f_{metric}_w', g=g_f, b=b_f,
                 X_all=x_runs_f, F_all=f_runs_f, T_all=t_runs_f, Acc_all=acc_runs_f, FT_all=ft_runs_f)
        print(f'Saving the final results to {folder}_{method}_{subset_size}_{rand}_best_acc_{metric}_w')
        np.savez(f'{folder}_{method}_{subset_size}_{rand}_best_acc_{metric}_w', g=g_a, b=b_a,
                 X_all=x_runs_a, F_all=f_runs_a, T_all=t_runs_a, Acc_all=acc_runs_a, FT_all=ft_runs_a)

    print('Finish')


def gradient_difference(data, method, rand, metric, reg=1e-5):
    folder = f'/tmp/{data}'
    train_data, train_target, val_data, val_target, test_data, test_target = load_dataset(data)

    num_runs = 1 if 'grd' in rand else 5
    subsets = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    max_diffs = np.zeros((num_runs, len(subsets)))
    max_full_grad_norms = np.zeros((num_runs, len(subsets)))

    for run in range(num_runs):
        for s in range(len(subsets)):
            subset_size = subsets[s]
            print(f'run {run}, gradient difference for subset: {subset_size}')
            if 'grd' in rand:
                file_name = glob.glob(f'{folder}_{subset_size}_{metric}.npz')[0]
                try:
                    dataset = np.load(f'{file_name}')
                    order, weights, total_ordering_time = dataset['order'], dataset['weight'], dataset['order_time']
                except:
                    print(f'could not read {file_name}')
                    continue
            else:
                print('Selecting a random subset')
                order = np.arange(0, len(train_data))
                random.shuffle(order)
                order = order[:int(subset_size * len(train_data))]
                weights = np.ones(len(order), dtype=np.float) * 1./subset_size

            try:
                res = np.load(f'{folder}_{method}_{subset_size}_{rand}_best_f_{metric}_w.npz', allow_pickle=True)
                non_empty = np.sum(res['F_all'], axis=1) > 0
                F, X = res['F_all'], res['X_all']
            except:
                print(f'could not read {folder}_{method}_{subset_size}_{rand}_best_f_{metric}_w.npz')
                continue

            best_run = np.argmin(F[non_empty, - 1])
            weights_all = np.ones(len(train_data))
            weights_all[order] = weights

            dim = len(train_data[0])
            num_class = 1 if data in ['covtype', 'ijcnn1'] else 3
            model = LogisticRegression(dim, num_class)

            max_diff, max_norm = 0, 0
            num_epochs = min(len(F[best_run]), 20)
            W = np.zeros((dim, num_class))  # initialize W 0
            b = np.zeros(num_class)  # initialize bias 0
            full_grad = [W, b]
            sub_grad = full_grad.copy()

            for epoch in range(num_epochs):
                model.params = X[best_run][epoch]
                #### random sample
                # W_sample = (np.random.rand(dim, num_class)*200-100) * np.ones((dim, num_class))
                # b_sample = (np.random.rand(num_class) * 200 - 100)
                # model.params = [W_sample, b_sample]
                #### random sample

                for i in range(len(train_data)):
                    grad = model.gradient(train_data[i], train_target[i], l2_reg=reg)
                    full_grad += grad
                    if i in order:
                        sub_grad += grad * weights_all[i]

                f_grad = np.append(np.reshape(full_grad[0], -1), full_grad[1])
                s_grad = np.append(np.reshape(sub_grad[0], -1), sub_grad[1])
                max_diff = max(max_diff, np.linalg.norm(f_grad - s_grad))
                max_norm = max(max_norm, np.linalg.norm(f_grad))

            max_diffs[run, s] = (max_diff / len(train_data))
            max_full_grad_norms[run, s] = (max_norm / len(train_data))
            print(*max_diffs, sep=', ')
            print(*max_full_grad_norms, sep=', ')
            tmp = 'rand_wgt' if 'rand_nw' in rand else rand
            print(f'Saving to {folder}_{method}_{tmp}_{metric}_grad_diff_w')
            np.savez(f'{folder}_{method}_{tmp}_{metric}_grad_diff_w', diff=max_diffs,
                     max_full_grad_norms=max_full_grad_norms)

    return max_diffs


if __name__ == '__main__':

    p = argparse.ArgumentParser(description='Faster Training.')
    p.add_argument('--data', type=str, required=False, default='covtype',
                   choices=['cifar10', 'covtype', 'mnist', 'ijcnn1', 'combined'], help='name of dataset')
    p.add_argument('--exp_decay', type=int, required=False, default=1,
                   choices=[0, 1], help='exponentially decaying learning rate')
    p.add_argument('--greedy', type=int, required=False, default=1,
                   help='greedy ordering')
    p.add_argument('--reg', type=float, required=False, default=1e-5,
                   help='L2 regularization constant')
    p.add_argument('--method', type=str, required=False, default='sgd',
                   choices=['sgd', 'svrg', 'saga'], help='sgd, svrg, saga')
    p.add_argument('--subset_size', type=float, required=False,
                   help='size of the subset')
    p.add_argument('--shuffle', type=int, default=2,
                   choices=[0, 1, 2, 3],
                   help='0: not shuffling, 1: random permutation, 2: with replacement, 3: fixed permutation')
    p.add_argument('--num_runs', type=int, required=False, default=10,
                   help='number of runs')
    p.add_argument('--metric', type=str, required=False, default='l2',
                   help='distance metric')
    p.add_argument('--b', type=float, required=False, default=-1,
                   help='learning rate parameter b')
    p.add_argument('--g', type=float, required=False, default=-1,
                   help='learning rate parameter g')
    p.add_argument('--num_epochs', type=int, required=False, default=-1,
                   help='number of epochs')
    p.add_argument('--grad_diff', type=int, required=False, default=0,
                   help='number of epochs')
    p.add_argument('--from_all', type=int, required=False, default=0,
                   help='number of epochs')

    args = p.parse_args()

    if args.greedy == 0:
        rand = 'rand_nw'
    elif args.greedy == 1 and args.shuffle == 1:
        rand = 'grd_shuff'
    elif args.greedy == 1 and args.shuffle == 2:
        rand = 'grd_rand'
    elif args.greedy == 1 and args.shuffle == 0:
        rand = 'grd_ord'
    elif args.greedy == 1 and args.shuffle > 2:
        rand = 'grd_fix_perm'
    else:
        rand = ''

    if args.grad_diff:
        gradient_difference(data=args.data, method=args.method, rand=rand, metric=args.metric)
    else:
        test(method=args.method, data=args.data, exp_decay=args.exp_decay, subset_size=args.subset_size,
             greedy=args.greedy, shuffle=args.shuffle, b_cnt=args.b, g_cnt=args.g, num_runs=args.num_runs,
             metric=args.metric, rand=rand, num_epochs=args.num_epochs, from_all=args.from_all)
