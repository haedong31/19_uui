from sklearn.neighbors import NearestNeighbors
import math
import numpy as np
import scipy.spatial as spatial


# epsilon: breakpoint contains more than 100 * r% of instances
def eps_vs(x, r, bin_size=10):
    # check validity of r
    if r > 1 or r < 0:
        print('[ERROR] r argument should be in [0, 1]')
        return  # terminate the function

    # get histogram data
    nbhd = NearestNeighbors(n_neighbors=2).fit(x)
    distances = nbhd.kneighbors(x)[0]
    dist2nn = [i[1] for i in distances]
    hist, bin_edges = np.histogram(dist2nn, bins=bin_size)

    # calculate epsilon
    cum_freq = 0
    for n, freq in enumerate(hist):
        cum_freq += freq
        if cum_freq / sum(hist) >= r:
            return bin_edges[n + 1]


# epsilon: weighted mean
def eps_wmean(x, bin_size=10):
    # get histogram data
    nbhd = NearestNeighbors(n_neighbors=2).fit(x)
    distances = nbhd.kneighbors(x)[0]
    dist2nn = [i[1] for i in distances]
    hist, bin_edges = np.histogram(dist2nn, bins=bin_size)

    # calculate epsilon
    bin_mid_pt = list()
    for n in range(len(hist)):
        left_pt = bin_edges[n]
        right_pt = bin_edges[n + 1]
        bin_mid_pt.append((left_pt + right_pt) / 2)

    return np.average(bin_mid_pt, weight=hist)


# minimum number of neighbors
def min_pt(x, eps, r, bin_size=10):
    # check validity of r
    if r > 1 or r < 0:
        print('[ERROR] r argument should be in [0, 1]')
        return  # terminate the function

    pt_tree = spatial.cKDTree(x)
    num_nbhds = list()
    for pt in x:
        num_nbhds.append(len(pt_tree.query_ball_point(pt, eps)))

    hist, bin_edges = np.histogram(num_nbhds, bins=bin_size)
    cum_freq = 0
    for n, freq in enumerate(hist):
        cum_freq += freq
        if 1 - (cum_freq / sum(hist)) <= r:
            return math.floor(bin_edges[n + 1])
