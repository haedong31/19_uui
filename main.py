from embeddings import *
from glob import glob
from search_param_dbscan import *
from sklearn.cluster import DBSCAN
import csv
import numpy as np
import os
import pickle
import timeit


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
            user_name = os.path.basename(file_paths[n]).split('_')[0]
            with open(os.path.join('.', 'data', 'bert_data', user_name + '_bert.txt'), 'wb') as f:
                pickle.dump(d, f)


def my_timer(fn, *args):
    start = timeit.default_timer()
    y = fn(*args)
    end = timeit.default_timer()
    print('[INFO] work time: {} sec'.format(end - start))

    return y


def my_evaluation(true_y, est_y, measure = 'accuracy'):
    true_y_set = list(set(true_y))
    est_y_set = list(set(est_y) - {-1})

    if measure == 'accuracy':
        c = np.empty([len(true_y_set), 2])
        result = np.empty([len(true_y_set), 4])  # true y, true n, est y, est n
        cluster_sizes = list()

        # find indices of true membership for each user
        for n, y in enumerate(true_y_set):
            user_idx = [n for n, x in enumerate(true_y) if x == y]
            cluster_sizes.append(len(user_idx))

            est_y_user = est_y[user_idx]
            tmp_labels, tmp_cnt = np.unique([x for x in est_y_user if x != -1], return_counts=True)
            c[n] = tmp_cnt
        label_assign = dict(zip(np.argmax(c, axis=0), est_y_set))

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
            result[n] = tmp_row
        return result
    else:
        print('enter proper performance measure')


# p = DataProcessor(glob('./data/collections_csv/*.csv'))
# p.csv_loader()
# p.bert()

# load the embedded data
file_paths = glob('./data/bert_data/*.txt')
sub_file_paths = np.random.choice(file_paths, 3)
bert_data = list()
for n, p in enumerate(sub_file_paths):
    print('### PROCESSING {} out of {}'.format(n+1, len(sub_file_paths)))
    with open(p, 'rb') as f:
        bert_data.append(pickle.load(f))

# turn the embedding matrix for a sentence into a row vector by averaging out
flat_bert_data = list()
true_labels = list()  # label vector
l = 0
for n, bd in enumerate(bert_data):
    print('### PROCESSING {} out of {}'.format(n + 1, len(bert_data)))
    for emd_mx in bd:
        flat_bert_data.append(emd_mx.mean(axis=0))
        true_labels.append(l)
    l += 1

# search the DBSCAN parameters
eps1 = my_timer(eps_vs, flat_bert_data, 0.8, 20)
eps2 = eps_wmean(flat_bert_data, 20)
min_samples1 = my_timer(min_pt, flat_bert_data, eps1, 0.8, 20)
# min_samples2 = min_pt(flat_bert_data, eps2, 0.85, 20)

# run DBSCAN
model = DBSCAN(eps=eps1, min_samples=min_samples1, metric='euclidean', n_jobs=2)

start = timeit.default_timer()
model.fit(flat_bert_data)
end = timeit.default_timer()

est_labels = model.labels_
n_clusters = len(set(est_labels)) - (1 if -1 in est_labels else 0)
n_noise = list(est_labels).count(-1)
my_evaluation(true_labels, est_labels)
