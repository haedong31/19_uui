from sklearn.neighbors import NearestNeighbors
import math
import numpy as np
import scipy.spatial as spatial


# epsilon: breakpoint contains more than 100 * r% of instances
def eps_vm(x, r, bin_size=10):
    num_bins = math.ceil(np.shape(x)[0] / bin_size)

    # check validity of r
    if r > 1 or r < 0:
        print('[ERROR] r argument should be in [0, 1]')
        return  # terminate the function

    # get histogram data
    nbhd = NearestNeighbors(n_neighbors=2).fit(x)
    distances = nbhd.kneighbors(x)[0]
    dist2nn = [i[1] for i in distances]
    hist, bin_edges = np.histogram(dist2nn, bins=num_bins)

    # calculate epsilon
    cum_freq = 0
    for n, freq in enumerate(hist):
        cum_freq += freq
        if cum_freq / sum(hist) >= r:
            return bin_edges[n + 1]


# epsilon: weighted mean
def eps_wmean(x, bin_size=10):
    num_bins = math.ceil(np.shape(x)[0] / bin_size)

    # get histogram data
    nbhd = NearestNeighbors(n_neighbors=2).fit(x)
    distances = nbhd.kneighbors(x)[0]
    dist2nn = [i[1] for i in distances]
    hist, bin_edges = np.histogram(dist2nn, bins=num_bins)

    # calculate epsilon
    bin_mid_pt = list()
    for n in range(len(hist)):
        left_pt = bin_edges[n]
        right_pt = bin_edges[n + 1]
        bin_mid_pt.append((left_pt + right_pt) / 2)

    return np.average(bin_mid_pt, weights=hist)


# nu: breakpoint contains more than 100 * r% of instances
def nu_vm(x, eps, r, bin_size=10):
    num_bins = math.ceil(np.shape(x)[0] / bin_size)

    # check validity of r
    if r > 1 or r < 0:
        print('[ERROR] r argument should be in [0, 1]')
        return  # terminate the function

    pt_tree = spatial.cKDTree(x)
    num_nbhds = list()
    for pt in x:
        num_nbhds.append(len(pt_tree.query_ball_point(pt, eps)))

    hist, bin_edges = np.histogram(num_nbhds, bins=num_bins)
    cum_freq = 0
    for n, freq in enumerate(hist):
        cum_freq += freq
        if 1 - (cum_freq / sum(hist)) <= r:
            return math.floor(bin_edges[n + 1])


# nu: weighted mean
def nu_wmean(x, eps, bin_size=10):
    num_bins = math.ceil(np.shape(x)[0] / bin_size)
    
    pt_tree = spatial.cKDTree(x)
    num_nbhds = list()
    for pt in x:
        num_nbhds.append(len(pt_tree.query_ball_point(pt, eps)))

    hist, bin_edges = np.histogram(num_nbhds, bins=num_bins)

    bin_mid_pt = list()
    for n in range(len(hist)):
        left_pt = bin_edges[n]
        right_pt = bin_edges[n + 1]
        bin_mid_pt.append((left_pt + right_pt) / 2)

    return np.average(bin_mid_pt, weights=hist)


def nu_wmean_trunc(x, eps, bin_size=10):
    num_bins = math.ceil(np.shape(x)[0] / bin_size)

    pt_tree = spatial.cKDTree(x)
    num_nbhds = list()
    for pt in x:
        num_nbhds.append(len(pt_tree.query_ball_point(pt, eps)))

    hist, bin_edges = np.histogram(num_nbhds, bins=num_bins)

    bin_mid_pt = list()
    for n in range(len(hist)):
        left_pt = bin_edges[n]
        right_pt = bin_edges[n + 1]
        bin_mid_pt.append((left_pt + right_pt) / 2)

    sorted_hist = -np.sort(-hist)  # descending order
    cutoff = 0
    while True:
        diff = (sorted_hist[cutoff] - sorted_hist[cutoff+1]) / 2
        if abs(diff) < 0.00001:
            break
        else:
            cutoff += 1

    top_pts_idx = [np.where(hist == x) for x in sorted_hist[range(cutoff)]]
    top_pts = [np.array(bin_mid_pt)[x] for x in top_pts_idx]

    numer = list()
    denom = list()
    for i in range(cutoff):
        tp = list(top_pts[i])
        sh = sorted_hist[i]

        numer.append(sum(tp) * len(tp) * sh)
        denom.append(len(tp) * sh)

    return sum(numer) / sum(denom)
