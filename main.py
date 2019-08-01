from glob import glob
from search_param_dbscan import *
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from utils import *
import csv
import numpy as np
import os
import pandas as pd
import pickle


class ExpProcessor(object):
    def __init__(self, n_users, ebd_type):
        self.n_users = n_users
        self.ebd_type = ebd_type
        self.ebd_data = list()
        self.true_y = list()
        self.cnt_table = None

    def file_loader(self):
        print('[INFO] loading embedding data files')
        file_paths = glob(os.path.join('.', 'data', self.ebd_type, '*.txt'))
        sub_file_paths = np.random.choice(file_paths, self.n_users)

        # load embedding data files
        for n, p in enumerate(sub_file_paths):
            print('### PROCESSING {} out of {}'.format(n+1, len(sub_file_paths)))
            with open(p, 'rb') as f:
                self.ebd_data.append(pickle.load(f))

        # create a ground-truth-label vector
        for n, d in enumerate(self.ebd_data):
            for i in range(len(d)):
                self.true_y.append(n)

    def mean_pooling(self):
        # turn the embedding matrix for a sentence into a row vector by averaging out
        print('[INFO] mean pooling of word embeddings')
        pool_ebd_data = list()
        for n, bd in enumerate(self.ebd_data):
            print('### PROCESSING {} out of {}'.format(n + 1, len(self.ebd_data)))
            for emd_mx in bd:
                pool_ebd_data.append(emd_mx.mean(axis=0))
        return pool_ebd_data

    def evaluation(self, est_y, alg, eps, nu):
        print('[INFO] evaluate performance')

        true_y_set = list(set(self.true_y))
        est_y_set = list(set(est_y) - {-1})
        cluster_sizes = list()
        self.cnt_table = np.empty([len(true_y_set), len(est_y_set)])
        result = pd.DataFrame(columns=['true_y', 'true_n', 'est_y', 'est_n'])

        # estimated membership and its count
        for n, y in enumerate(true_y_set):
            tmp_freq = dict(zip(est_y_set, [0] * len(est_y_set)))

            user_idx = [n for n, x in enumerate(e.true_y) if x == y]
            cluster_sizes.append(len(user_idx))
            est_y_user = est_y[user_idx]

            tmp_y, tmp_cnt = np.unique([x for x in est_y_user if x != -1], return_counts=True)
            for m, _y in enumerate(list(tmp_y)):
                tmp_freq[_y] = tmp_cnt[m]

            self.cnt_table[n] = list(tmp_freq.values())

        # label assignment
        label_assign = dict()  # key (user) : value (estimated membership)
        dummy_c = self.cnt_table
        n_rows = np.shape(self.cnt_table)[0]
        n_cols = np.shape(self.cnt_table)[1]
        i = 0
        while i < n_rows and i < n_cols:
            max_n = np.amax(dummy_c)

            # assign estimated cluster membership to a user
            max_idx = np.where(self.cnt_table == max_n)
            max_idx_row = max_idx[0]
            max_idx_col = max_idx[1]

            # make sure max_idx only contains indices of row and column that never have been selected
            for r, c in label_assign.items():
                # row wise
                tmp_del_idx = np.where(max_idx_row==r)
                max_idx_row = np.delete(max_idx_row, tmp_del_idx)
                max_idx_col = np.delete(max_idx_col, tmp_del_idx)

                # column wise
                tmp_del_idx = np.where(max_idx_col==c)
                max_idx_row = np.delete(max_idx_row, tmp_del_idx)
                max_idx_col = np.delete(max_idx_col, tmp_del_idx)

            if len(max_idx_row) != len(max_idx_col):
                print('[ERROR] lengths of max_idx_row and max_idx_col do not match')
                return

            # handle the case of tie; break the tie by random choice
            if len(max_idx_row) != 1:
                tie_idx = np.random.choice(range(len(max_idx_row)))
                label_assign.update({int(max_idx_row[tie_idx]): int(max_idx_col[tie_idx])})
            else:
                label_assign.update({int(max_idx_row): int(max_idx_col)})

            # delete already got assigned cluster label and user from c
            max_idx_row, max_idx_col = list(label_assign.items())[-1]
            adj_row = sum([1 for x in label_assign.keys() if x < max_idx_row])
            adj_col = sum([1 for x in label_assign.values() if x < max_idx_col])

            dummy_max_idx_row = max_idx_row - adj_row
            dummy_max_idx_col = max_idx_col - adj_col

            dummy_c = np.delete(dummy_c, dummy_max_idx_row, axis=0)
            dummy_c = np.delete(dummy_c, dummy_max_idx_col, axis=1)

            i += 1

        # summarize in a Pandas data frame
        for n, y in enumerate(true_y_set):
            tmp_row = list()
            tmp_row.append(y)
            tmp_row.append(cluster_sizes[n])
            if y in label_assign.keys():
                tmp_row.append(label_assign[y])
                tmp_row.append(self.cnt_table[y][label_assign[y]])
            else:
                tmp_row.append(-1)
                tmp_row.append(0)
            result = result.append(pd.Series(tmp_row, index=result.columns), ignore_index=True)
        print(result)

        # additional information
        # accuracy
        if any(result['est_n'] - result['true_n'] > 0):
            n_crr = list()
            for n, row in result.iterrows():
                if row['est_n'] > row['true_y']:
                    n_crr[n] = row['true_y']
                else:
                    n_crr[n] = row['est_y']
            accuracy = sum(n_crr) / len(self.true_y)
        else:
            accuracy = sum(result['est_n']) / len(self.true_y)
        print('accuracy: {}'.format(accuracy))

        # number of estimated users and noises
        n_clusters = len(set(est_y)) - (1 if -1 in est_y else 0)
        n_noise = list(est_y).count(-1)
        print('number of estimated users: {} / noises: {}'.format(n_clusters, n_noise))

        # create a saving directory
        save_dir = os.path.join('.', 'result', alg)
        os.makedirs(save_dir, exist_ok=True)

        # save an evaluation table
        file_name = f'{len(true_y_set)}_{eps:.2f}_{nu}.csv'
        save_path = os.path.join(save_dir, file_name + '.csv')
        result.to_csv(save_path)

        # add additional information
        print('save the result')
        with open(save_path, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['accuracy', 'n_users_est', 'n_noise', 'eps', 'min_pts'])
            csv_writer.writerow([accuracy, n_clusters, n_noise, eps, nu])


e = ExpProcessor(3, 'bert')
e.file_loader()
flat_bert = e.mean_pooling()

# apply PCA
pca_model = PCA(n_components=128)
pca_flat_bert = pca_model.fit_transform(flat_bert)
sum(pca_model.explained_variance_ratio_)

# search the DBSCAN parameters
# eps = my_timer(eps_vs, flat_bert, 0.8, 20)
eps = my_timer(eps_wmean, pca_flat_bert)
nu1 = my_timer(nu_wmean, pca_flat_bert, eps)
nu2 = my_timer(nu_wmean_trunc, pca_flat_bert, eps)
nu = (nu1 + nu2) / 2
print(eps)
print(nu)

# run DBSCAN
dbscan_model = DBSCAN(eps=eps, min_samples=nu, metric='euclidean', n_jobs=2)
start = timeit.default_timer()
dbscan_model.fit(pca_flat_bert)
est_labels = dbscan_model.labels_
end = timeit.default_timer()
print('[INFO] work time fitting DBSCAN: {} min'.format((end - start) / 60))

e.evaluation(est_labels, 'bert', eps, nu)
