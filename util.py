import os
import subprocess
import time
import gc

from nearpy import Engine
from nearpy.distances import EuclideanDistance
from nearpy.filters import NearestFilter
from nearpy.hashes import RandomBinaryProjections
import numpy as np
from lazy_greedy import FacilityLocation, lazy_greedy_heap
import sklearn

from tensorflow.examples.tutorials.mnist import input_data

SEED = 100
EPS = 1E-8
PLOT_NAMES = ['lr', 'data_loss', 'epoch_loss', 'test_loss']  # 'cluster_compare', 'cosine_compare', 'euclidean_compare'


def load_dataset(dataset, dataset_dir):
    '''
    Args
    - dataset: str, one of ['cifar10', 'covtype'] or filename in `data/`
    - dataset_dir: str, path to `data` folder

    Returns
    - X: np.array, shape [N, d]
      - exception: shape [N, 32, 32, 3] for cifar10
    - y: np.array, shape [N]
    '''
    if dataset == 'cifar10':
        path = os.path.join(dataset_dir, 'cifar10', 'cifar10.npz')
        with np.load(path) as npz:
            X = npz['x']  # shape [60000, 32, 32, 3], type uint8
            y = npz['y']  # shape [60000], type uint8
        # convert to float in (0, 1), center at mean 0
        X = X.astype(np.float32) / 255
        # X -= np.mean(X, axis=0)
    elif dataset == 'cifar10_features':
        path = os.path.join(dataset_dir, 'cifar10', 'train_features.npz')
        with np.load(path) as npz:
            X = npz['features']  # shape [50000, 64], type float32
            y = npz['labels']  # shape [50000], type int64
    elif dataset == 'cifar10_grads':
        # labels
        path = os.path.join(dataset_dir, 'cifar10', 'train_features.npz')
        with np.load(path) as npz:
            y = npz['labels']  # shape [50000], type int64
        # feautres
        path = os.path.join('grad_features.npy')
        X = np.load(path)  # shape [50000, 1000], type float16
    elif dataset == 'mnist':
        mnist = input_data.read_data_sets('/tmp')
        X_train = np.vstack([mnist.train.images, mnist.validation.images])
        y_train = np.hstack([mnist.train.labels, mnist.validation.labels])
        X_test = mnist.test.images
        y_test = mnist.test.labels
        X_train = X_train.astype(np.float32) / 255
        X_test = X_test.astype(np.float32) / 255
        return X_train, y_train, X_test, y_test

    else:
        num, dim, name = 0, 0, ''
        if dataset == 'covtype':
            num, dim = 581012, 54
            name = 'covtype.libsvm.binary.scale'
        elif dataset == 'ijcnn1.t' or dataset == 'ijcnn1.tr':
            num, dim = 49990 if 'tr' in dataset else 91701, 22
            name = dataset
        elif dataset == 'combined_scale' or dataset == 'combined_scale.t':
            num, dim = 19705 if '.t' in dataset else 78823, 100
            name = dataset

        X = np.zeros((num, dim), dtype=np.float32)
        y = np.zeros(num, dtype=np.int32)
        path = os.path.join(dataset_dir, name)

        with open(path, 'r') as f:
            for i, line in enumerate(f):
                y[i] = float(line.split()[0])
                for e in line.split()[1:]:
                    cur = e.split(':')
                    X[i][int(cur[0]) - 1] = float(cur[1])
                i += 1
        y = np.array(y, dtype=np.int32)
        if name in ['ijcnn1.t', 'ijcnn1.tr']:
            y[y == -1] = 0
        else:
            y = y - np.ones(len(y), dtype=np.int32)

    return X, y


def compute_raw_clusters(X):
    '''
    Args
    - X: np.array, shape [N, d]

    Returns
    - clusters: list of lists of indices
      - clusters[c] gives the indices into X corresponding to cluster c
      - clusters are sorted from largest to smallest
    '''
    print('Computing raw clusters...')
    N, d = X.shape
    start = time.time()
    rbp = RandomBinaryProjections('rbp', 10, rand_seed=SEED)
    engine = Engine(d, lshashes=[rbp], distance=EuclideanDistance(), vector_filters=[NearestFilter(1)])
    cluster_idx = -1 * np.ones(N)
    clusters = []
    for i in range(N):
        neighbor_list = engine.neighbours(X[i])
        if len(neighbor_list) > 0:
            neighbor = neighbor_list[0][1]
            cluster_idx[i] = cluster_idx[neighbor]
            clusters[cluster_idx[i]].append(i)
        else:
            cluster_idx[i] = len(clusters)
            clusters.append([i])
        engine.store_vector(X[i], data=i)
        if (i + 1) % 10000 == 0:
            print(f'finished {i + 1} points')

    print('number of clusters:', len(clusters))
    print('time:', time.time() - start)
    return sorted(clusters, key=lambda x: -len(x))


def similarity(X, metric):
    '''Computes the similarity between each pair of examples in X.

    Args
    - X: np.array, shape [N, d]
    - metric: str, one of ['cosine', 'euclidean']

    Returns
    - S: np.array, shape [N, N]
    '''
    start = time.time()
    dists = sklearn.metrics.pairwise_distances(X, metric=metric, n_jobs=1)
    elapsed = time.time() - start

    if metric == 'cosine':
        S = 1 - dists
    elif metric == 'euclidean' or metric == 'l1':
        m = np.max(dists)
        S = m - dists
    else:
        raise ValueError(f'unknown metric: {metric}')

    return S, elapsed


def greedy_merge(X, y, B, part_num, metric, smtk=0, stoch_greedy=False):
    N = len(X)
    indices = list(range(N))
    # np.random.shuffle(indices)
    part_size = int(np.ceil(N / part_num))
    part_indices = [indices[slice(i * part_size, min((i + 1) * part_size, N))] for i in range(part_num)]
    print(f'GreeDi with {part_num} parts, finding {B} elements...', flush=True)

    # pool = ThreadPool(part_num)
    # order_mg_all, cluster_sizes_all, _, _, ordering_time, similarity_time = zip(*pool.map(
    #     lambda p: get_orders_and_weights(
    #         int(B / 2), X[part_indices[p], :], metric, p + 1, stoch_greedy, y[part_indices[p]]), np.arange(part_num)))
    # pool.terminate()

    order_mg_all, cluster_sizes_all, _, _, ordering_time, similarity_time, F_val = zip(*map(
        lambda p: get_orders_and_weights(
            int(B / 2), X[part_indices[p], :], metric, p + 1, stoch_greedy, y[part_indices[p]]), np.arange(part_num)))

    # Returns the number of objects it has collected and deallocated
    collected = gc.collect()
    print(f'Garbage collector: collected {collected}')

    # order_mg_all = np.zeros((part_num, B))
    # cluster_sizes_all = np.zeros((part_num, B))
    # ordering_time = np.zeros(part_num)
    # similarity_time = np.zeros(part_num)
    # for p in range(part_num):
    #    order_mg_all[p,:], cluster_sizes_all[p,:], _, _, ordering_time[p], similarity_time[p] = get_orders_and_weights(
    #         B, X[part_indices[p], :], metric, p, stoch_greedy, y[part_indices[p]])
    order_mg_all = np.array(order_mg_all, dtype=np.int32)
    cluster_sizes_all = np.array(cluster_sizes_all, dtype=np.float32)  # / part_num (not needed)
    order_mg = order_mg_all.flatten(order='F')
    weights_mg = cluster_sizes_all.flatten(order='F')
    print(f'GreeDi stage 1: found {len(order_mg)} elements in: {np.max(ordering_time)} sec', flush=True)

    # order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    order, weights, order_sz, weights_sz, ordering_time_merge, similarity_time_merge = get_orders_and_weights(
        B, X[order_mg, :], metric, smtk, stoch_greedy, y[order_mg], weights_mg)
    print(weights)
    total_ordering_time = np.max(ordering_time) + ordering_time_merge
    total_similarity_time = np.max(similarity_time) + similarity_time_merge
    print(f'GreeDi stage 2: found {len(order)} elements in: {total_ordering_time + total_similarity_time} sec',
          flush=True)
    vals = order, weights, order_sz, weights_sz, total_ordering_time, total_similarity_time
    return vals


def greedi(X, y, B, part_num, metric, smtk=0, stoch_greedy=False, seed=-1):
    N = len(X)
    indices = list(range(N))
    if seed != -1:
        np.random.seed(seed)
        np.random.shuffle(indices)  # Note: random shuffling
    part_size = int(np.ceil(N / part_num))
    part_indices = [indices[slice(i * part_size, min((i + 1) * part_size, N))] for i in range(part_num)]
    print(f'GreeDi with {part_num} parts, finding {B} elements...', flush=True)

    # pool = ThreadPool(part_num)
    # order_mg_all, cluster_sizes_all, _, _, ordering_time, similarity_time = zip(*pool.map(
    #     lambda p: get_orders_and_weights(
    #         B, X[part_indices[p], :], metric, p + 1, stoch_greedy, y[part_indices[p]]), np.arange(part_num)))
    # pool.terminate()
    # Returns the number of objects it has collected and deallocated
    # collected = gc.collect()
    # print(f'Garbage collector: collected {collected}')
    order_mg_all, cluster_sizes_all, _, _, ordering_time, similarity_time = zip(*map(
        lambda p: get_orders_and_weights(
            B, X[part_indices[p], :], metric, p + 1, stoch_greedy, y[part_indices[p]]), np.arange(part_num)))
    gc.collect()

    order_mg_all = np.array(order_mg_all, dtype=np.int32)
    for c in np.arange(part_num):
        order_mg_all[c] = np.array(part_indices[c])[order_mg_all[c]]
    # order_mg_all = np.zeros((part_num, B))
    # cluster_sizes_all = np.zeros((part_num, B))
    # ordering_time = np.zeros(part_num)
    # similarity_time = np.zeros(part_num)
    # for p in range(part_num):
    #    order_mg_all[p,:], cluster_sizes_all[p,:], _, _, ordering_time[p], similarity_time[p] = get_orders_and_weights(
    #         B, X[part_indices[p], :], metric, p, stoch_greedy, y[part_indices[p]])
    cluster_sizes_all = np.array(cluster_sizes_all, dtype=np.float32)  # / part_num (not needed)
    order_mg = order_mg_all.flatten(order='F')
    weights_mg = cluster_sizes_all.flatten(order='F')
    print(f'GreeDi stage 1: found {len(order_mg)} elements in: {np.max(ordering_time)} sec', flush=True)

    # order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    order, weights, order_sz, weights_sz, ordering_time_merge, similarity_time_merge = get_orders_and_weights(
        B, X[order_mg,:], metric, smtk, stoch_greedy, y[order_mg], weights_mg)
    print(weights)
    order = order_mg[order]
    total_ordering_time = np.max(ordering_time) + ordering_time_merge
    total_similarity_time = np.max(similarity_time) + similarity_time_merge
    print(f'GreeDi stage 2: found {len(order)} elements in: {total_ordering_time + total_similarity_time} sec', flush=True)
    vals = order, weights, order_sz, weights_sz, total_ordering_time, total_similarity_time
    return vals


def get_facility_location_submodular_order(S, B, c, smtk=0, no=0, stoch_greedy=0, weights=None):
    '''
    Args
    - S: np.array, shape [N, N], similarity matrix
    - B: int, number of points to select

    Returns
    - order: np.array, shape [B], order of points selected by facility location
    - sz: np.array, shape [B], type int64, size of cluster associated with each selected point
    '''
    # print('Computing facility location submodular order...')
    N = S.shape[0]
    no = smtk if no == 0 else no

    if smtk > 0:
        print(f'Calculating ordering with SMTK... part size: {len(S)}, B: {B}', flush=True)
        np.save(f'/tmp/{no}/{smtk}-{c}', S)
        if stoch_greedy > 0:
            p = subprocess.check_output(
                f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                 -stochastic-greedy -sg-epsilon {stoch_greedy} -flnpy /tmp/{no}/{smtk}-{c}.'
                f'npy -pnpv -porder -ptime'.split())
        else:
            p = subprocess.check_output(
                f'/tmp/{no}/smtk-master{smtk}/build/smraiz -sumsize {B} \
                             -flnpy /tmp/{no}/{smtk}-{c}.npy -pnpv -porder -ptime'.split())
        s = p.decode("utf-8")
        str, end = ['([', ',])']
        order = s[s.find(str) + len(str):s.rfind(end)].split(',')
        greedy_time = float(s[s.find('CPU') + 4 : s.find('s (User')])
        # print(f'FL greedy time: {greedy_time}', flush=True)
        str = 'f(Solution) = '
        F_val = float(s[s.find(str) + len(str) : s.find('Summary Solution') - 1])
        # print(f'===========> f(Solution) = {F_val}')
    else:
        V = list(range(N))
        start = time.time()
        F = FacilityLocation(S, V)
        order, _ = lazy_greedy_heap(F, V, B)
        greedy_time = time.time() - start
        F_val = 0 # TODO

    order = np.asarray(order, dtype=np.int64)
    sz = np.zeros(B, dtype=np.float64)
    for i in range(N):
        if weights is None:
            sz[np.argmax(S[i, order])] += 1
        else:
            sz[np.argmax(S[i, order])] += weights[i]
    collected = gc.collect()
    return order, sz, greedy_time, F_val


def faciliy_location_order(c, X, y, metric, num_per_class, smtk, no, stoch_greedy, weights=None):
    class_indices = np.where(y == c)[0]
    print(c)
    print(class_indices)
    print(len(class_indices))
    S, S_time = similarity(X[class_indices], metric=metric)
    order, cluster_sz, greedy_time, F_val = get_facility_location_submodular_order(
        S, num_per_class, c, smtk, no, stoch_greedy, weights)
    return class_indices[order], cluster_sz, greedy_time, S_time


def get_orders_and_weights(B, X, metric, smtk, no=0, stoch_greedy=0, y=None, weights=None, equal_num=False, outdir='.'): #todo
    '''
    Ags
    - X: np.array, shape [N, d]
    - B: int, number of points to select
    - metric: str, one of ['cosine', 'euclidean'], for similarity
    - y: np.array, shape [N], integer class labels for C classes
      - if given, chooses B / C points per class, B must be divisible by C
    - outdir: str, path to output directory, must already exist

    Returns
    - order_mg/_sz: np.array, shape [B], type int64
      - *_mg: order points by their marginal gain in FL objective (largest gain first)
      - *_sz: order points by their cluster size (largest size first)
    - weights_mg/_sz: np.array, shape [B], type float32, sums to 1
    '''
    N = X.shape[0]
    if y is None:
        y = np.zeros(N, dtype=np.int32)  # assign every point to the same class
    classes = np.unique(y)
    C = len(classes)  # number of classes
    # assert np.array_equal(classes, np.arange(C))
    # assert B % C == 0

    if equal_num:
        class_nums = [sum(y == c) for c in classes]
        num_per_class = int(np.ceil(B / C)) * np.ones(len(classes), dtype=np.int32)
        minority = class_nums < np.ceil(B / C)
        if sum(minority) > 0:
            extra = sum([max(0, np.ceil(B / C) - class_nums[c]) for c in classes])
            for c in classes[~minority]:
                num_per_class[c] += int(np.ceil(extra / sum(minority)))
    else:
        num_per_class = np.int32(np.ceil(np.divide([sum(y == i) for i in classes], N) * B))
        print('not equal_num')

    # print(f'Greedy: selecting {num_per_class} elements')

    # order_mg_all = np.zeros([C, num_per_class], dtype=np.int64)
    # cluster_sizes_all = np.zeros([C, num_per_class], dtype=np.float32)
    # greedy_time_all = np.zeros([C, num_per_class], dtype=np.int64)
    # similarity_time_all = np.zeros([C, num_per_class], dtype=np.int64)

    # pool = ThreadPool(C)
    # order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(*pool.map(
    #     lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], smtk, stoch_greedy, weights), classes))
    # pool.terminate()
    order_mg_all, cluster_sizes_all, greedy_times, similarity_times = zip(*map(
        lambda c: faciliy_location_order(c, X, y, metric, num_per_class[c], smtk, no, stoch_greedy, weights), classes))

    order_mg, weights_mg = [], []
    if equal_num:
        props = np.rint([len(order_mg_all[i]) for i in range(len(order_mg_all))])
    else:
        # merging imbalanced classes
        class_ratios = np.divide([np.sum(y == i) for i in classes], N)
        props = np.rint(class_ratios / np.min(class_ratios)) # TODO
        print(f'Selecting with ratios {np.array(class_ratios)}')
        print(f'Class proportions {np.array(props)}')

    order_mg_all = np.array(order_mg_all)
    cluster_sizes_all = np.array(cluster_sizes_all)
    for i in range(int(np.rint(np.max([len(order_mg_all[c]) / props[c] for c in classes])))):
        for c in classes:
            ndx = slice(i * int(props[c]), int(min(len(order_mg_all[c]), (i + 1) * props[c])))
            order_mg = np.append(order_mg, order_mg_all[c][ndx])
            weights_mg = np.append(weights_mg, cluster_sizes_all[c][ndx])
    order_mg = np.array(order_mg, dtype=np.int32)

    # class_ratios = np.divide([np.sum(y == i) for i in classes], N)
    # weights_mg[y[order_mg] == np.argmax(class_ratios)] /= (np.max(class_ratios) / np.min(class_ratios))

    weights_mg = np.array(weights_mg, dtype=np.float32)
    ordering_time = np.max(greedy_times)
    similarity_time = np.max(similarity_times)

    # for c in classes:
    #     class_indices = np.where(y == c)[0]
    #     S, similarity_time_all[c] = similarity(X[class_indices], metric=metric)
    #     order, cluster_sz, greedy_time_all[c], F_val = get_facility_location_submodular_order(S, num_per_class, c, smtk)
    #     order_mg_all[c] = class_indices[order]
    #     cluster_sizes_all[c] = cluster_sz
    #     save_cluster_sizes(cluster_sizes_all[c], metric=f'{metric}_class{c}', outdir=outdir)
    # cluster_sizes_all /= N

    # choose 1st from each class, then 2nd from each class, etc.
    # i.e. column-major order
    # order_mg_all = np.array(order_mg_all)
    # cluster_sizes_all = np.array(cluster_sizes_all, dtype=np.float32) / N
    # order_mg = order_mg_all.flatten(order='F')
    # weights_mg = cluster_sizes_all.flatten(order='F')

    # sort by descending cluster size within each class
    # cluster_order = np.argsort(-cluster_sizes_all, axis=1)
    # rows_selector = np.arange(C)[:, np.newaxis]
    order_sz = []  # order_mg_all[rows_selector, cluster_order].flatten(order='F')
    weights_sz = [] # cluster_sizes_all[rows_selector, cluster_order].flatten(order='F')
    vals = order_mg, weights_mg, order_sz, weights_sz, ordering_time, similarity_time
    return vals
