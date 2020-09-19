print(__doc__)
import matplotlib
#matplotlib.use('TkAgg')

import heapq
import numpy as np
import pandas as pd
import scipy as sp
import math
from scipy import spatial
import matplotlib.pyplot as plt


class FacilityLocation:

    def __init__(self, D, V, alpha=1.):
        '''
        Args
        - D: np.array, shape [N, N], similarity matrix
        - V: list of int, indices of columns of D
        - alpha: float
        '''
        self.D = D
        self.curVal = 0
        self.curMax = np.zeros(len(D))
        self.gains = []
        self.alpha = alpha
        self.f_norm = self.alpha / self.f_norm(V)
        self.norm = 1. / self.inc(V, [])

    def f_norm(self, sset):
        return self.D[:, sset].max(axis=1).sum()

    def inc(self, sset, ndx):
        if len(sset + [ndx]) > 1:
            if not ndx:  # normalization
                return math.log(1 + self.alpha * 1)
            return self.norm * math.log(1 + self.f_norm * np.maximum(self.curMax, self.D[:, ndx]).sum()) - self.curVal
        else:
            return self.norm * math.log(1 + self.f_norm * self.D[:, ndx].sum()) - self.curVal

    def add(self, sset, ndx):
        cur_old = self.curVal
        if len(sset + [ndx]) > 1:
            self.curMax = np.maximum(self.curMax, self.D[:, ndx])
        else:
            self.curMax = self.D[:, ndx]
        self.curVal = self.norm * math.log(1 + self.f_norm * self.curMax.sum())
        self.gains.extend([self.curVal - cur_old])
        return self.curVal


def lazy_greedy(F, ndx, B):
    '''
    Args
    - F: FacilityLocation
    - ndx: indices of all points
    - B: int, number of points to select
    '''
    TOL = 1e-6
    eps = 1e-15
    curVal = 0
    sset = []
    order = []
    vals = []
    for v in ndx:
        marginal = F.inc(sset, v) + eps
        heapq.heappush(order, (1.0 / marginal, v, marginal))

    while order and len(sset) < B:
        el = heapq.heappop(order)
        if not sset:
            improv = el[2]
        else:
            improv = F.inc(sset, el[1]) + eps

        # check for uniques elements
        if improv > 0 + eps:
            if not order:
                curVal = F.add(sset, el[1])
                # print curVal
                #print str(len(sset)) + ', ' + str(el[1])
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = heapq.heappop(order)
                if improv >= top[2]:
                    curVal = F.add(sset, el[1])
                    #print curVal
                    #print str(len(sset)) + ', ' + str(el[1])
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    heapq.heappush(order, (1.0 / improv, el[1], improv))
                heapq.heappush(order, top)
        else:
            2

    #print(sset)
    return sset, vals


def _heappush_max(heap, item):
    heap.append(item)
    heapq._siftdown_max(heap, 0, len(heap)-1)


def _heappop_max(heap):
    """Maxheap version of a heappop."""
    lastelt = heap.pop()  # raises appropriate IndexError if heap is empty
    if heap:
        returnitem = heap[0]
        heap[0] = lastelt
        heapq._siftup_max(heap, 0)
        return returnitem
    return lastelt


def lazy_greedy_heap(F, V, B):
    curVal = 0
    sset = []
    vals = []

    order = []
    heapq._heapify_max(order)
    [_heappush_max(order, (F.inc(sset, index), index)) for index in V]

    while order and len(sset) < B:
        el = _heappop_max(order)
        improv = F.inc(sset, el[1])

        # check for uniques elements
        if improv >= 0: #TODO <====
            if not order:
                curVal = F.add(sset, el[1])
                # print curVal
                sset.append(el[1])
                vals.append(curVal)
            else:
                top = _heappop_max(order)
                if improv >= top[0]:
                    curVal = F.add(sset, el[1])
                    # print curVal
                    sset.append(el[1])
                    vals.append(curVal)
                else:
                    _heappush_max(order, (improv, el[1]))
                _heappush_max(order, top)

    #print(str(sset) + ', val: ' + str(curVal))

    return sset, vals


def test():
    n = 10
    X = np.random.rand(n, n)
    D = X * np.transpose(X)
    F = FacilityLocation(D, range(0, n))
    sset = lazy_greedy(F, xrange(0, n), 15)
    print(sset)


def cifar(B, num_data):
    G = pd.read_csv('/Users/baharan/Downloads/cifar10/resnet20/1533688447/feats.csv', nrows=num_data).values
    n, dimensions = G.shape
    mymean = np.mean(G, axis=1)
    G = G - np.reshape(mymean, (n, 1)) * np.ones((1, dimensions))
    mynorm = np.linalg.norm(G, axis=1)
    N = np.matmul(np.reshape(mynorm, (n, 1)), np.ones((1, dimensions)))
    G = G / N
    G[np.argwhere(np.isnan(G))] = 0
    G = G + 3 * np.ones((n, dimensions)) / np.sqrt(dimensions)  # shift away from origin
    D = spatial.distance.cdist(G, G, 'euclidean')
    N = np.linalg.norm(G, axis=1)
    N = np.ones((1, n)) * np.reshape(N, (n, 1))
    D = N - D
    F = FacilityLocation(D, range(0, n))
    sset, vals = lazy_greedy(F, xrange(0, n), B)
    print(sset)
    plt.plot(vals)
    plt.show()
    plt.savefig('cifar10.png')


# cifar(500, 50)
#test()
