from embeddings import *
from glob import glob
from search_param_dbscan import *
from sklearn.cluster import DBSCAN
import csv
import numpy as np
import os
import pandas as pd
import pickle
import timeit


def my_timer(fn, *args):
    start = timeit.default_timer()
    y = fn(*args)
    end = timeit.default_timer()
    print('[INFO] work time: {} min'.format((end - start) / 60))

    return y


class DataProcessor(object):
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.raw_data = list()

    def csv_loader(self):
        print('[INFO] load csv files')

        num_files = len(self.file_paths)
        for i in range(num_files):
            print('### PROCESSING {} out of {}'.format(i+1, num_files))

            with open(self.file_paths[i], 'rt') as f:
                csv_reader = csv.reader(f)
                next(csv_reader)  # skip the header
                itr_list = list()
                for r in csv_reader:
                    itr_list.append(r[2])

            self.raw_data.append(itr_list)

    def bert(self):
        if not self.raw_data:
            print('[ERROR] load raw data first')
            return

        # embed the raw data with BERT
        print('[INFO] embed the raw data with BERT')
        num_datasets = len(self.raw_data)
        bert_data = list()
        bert_model = load_bert_model()
        for i in range(num_datasets):
            print('### PROCESSING {} out of {}'.format(i+1, num_datasets))
            bert_data.append(generate_vecs_bert(bert_model, self.raw_data[i], type='matrix'))

        # create a saving directory
        save_path = './data/bert_data'
        if os.path.exists('./data/bert_data'):
            cnt = 1
            while True:
                save_path = os.path.join('.', 'data', 'bert_data_' + cnt)
                if os.path.exists(save_path):
                    cnt += 1
                else:
                    break
        os.makedirs(save_path, exist_ok=False)

        # save the embedded data in the pickle format
        print('[INFO] save the BERT-embedded data into {}'.format(save_path))
        for n, d in enumerate(bert_data):
            print('### PROCESSING {} out of {}'.format(n + 1, num_datasets))
            user_name = os.path.basename(self.file_paths[n]).split('_')[0]
            with open(os.path.join('.', 'data', 'bert_data', user_name + '_bert.txt'), 'wb') as f:
                pickle.dump(d, f)


class ExpProcessor(object):
    def __init__(self, n_users, ebd_type):
        self.n_users = n_users
        self.ebd_type = ebd_type
        self.ebd_data = list()
        self.true_y = list()

    def file_loader(self):
        print('[INFO] loading embedding data files')
        file_paths = glob(os.path.join('.', 'data', self.ebd_type, '*.txt'))
        sub_file_paths = np.random.choice(file_paths, self.n_users)
        
        # load embedding data files
        for n, p in enumerate(sub_file_paths):
            print('### PROCESSING {} out of {}'.format(n+1, len(sub_file_paths)))
            with open(p, 'rb') as f:
                self.ebd_data.append(pickle.load(f))
        
        # create a gounrd-truth-label vector
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

    def evaluation(self, est_y, alg, eps, min_pts):
        print('[INFO] evaluate performance')

        true_y_set = list(set(self.true_y))
        est_y_set = list(set(est_y) - {-1})
        cluster_sizes = list()
        c = np.empty([len(true_y_set), len(est_y_set)])
        result = pd.DataFrame(columns=['true_y', 'true_n', 'est_y', 'est_n'])

        # estimated membership and its count
        for n, y in enumerate(true_y_set):
            user_idx = [n for n, x in enumerate(self.true_y) if x == y]
            cluster_sizes.append(len(user_idx))
            est_y_user = est_y[user_idx]
            tmp_cnt = np.unique([x for x in est_y_user if x != -1], return_counts=True)[1]
            c[n] = tmp_cnt
        label_assign = dict(zip(np.argmax(c, axis=0), est_y_set))

        # summarize in a Pandas data frame
        for n, y in enumerate(true_y_set):
            tmp_row = list()
            tmp_row.append(y)
            tmp_row.append(cluster_sizes[n])
            if y in label_assign.keys():
                tmp_row.append(label_assign[y])
                tmp_row.append(c[y][label_assign[y]])
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
        file_name = f'{len(true_y_set)}_{eps:.2f}_{min_samples}.csv'
        save_path = os.path.join(save_dir, file_name + '.csv')
        result.to_csv(save_path)
        
        # add additional information
        print('save the result')
        with open(save_path, 'a') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['accuracy', 'n_users_est', 'n_noise', 'eps', 'min_pts'])
            csv_writer.writerow([accuracy, n_clusters, n_noise, eps, min_pts])


# d = DataProcessor(glob('./data/collections_csv/*.csv'))
# d.csv_loader()
# d.bert()

e = ExpProcessor(3, 'bert')
e.file_loader()
flat_bert_data = e.mean_pooling()

# search the DBSCAN parameters
# eps = my_timer(eps_vs, flat_bert_data, 0.8, 20)
eps = my_timer(eps_wmean, flat_bert_data, 20)
min_samples = my_timer(min_pt, flat_bert_data, eps, 0.8, 20)

# run DBSCAN
model = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', n_jobs=2)

start = timeit.default_timer()
model.fit(flat_bert_data)
est_labels = model.labels_
end = timeit.default_timer()
print('[INFO] work time: {} min'.format((end - start) / 60))

n_clusters = len(set(est_labels)) - (1 if -1 in est_labels else 0)
n_noise = list(est_labels).count(-1)

e.evaluation(est_labels, 'bert', eps, min_samples)
