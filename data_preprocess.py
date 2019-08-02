from embeddings import *
from glob import glob
import csv
import os
import pickle


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

    def pickle_loader(self):
        print('[INFO] load pickle files')

        num_files = len(self.file_paths)
        for i in range(num_files):
            print('### PROCESSING {} out of {}'.format(i + 1, num_files))

            with open(self.file_paths[i], 'rb') as f:
                self.raw_data.append(pickle.load(f))

    def bert(self):
        if not self.raw_data:
            print('[ERROR] load raw data first')
            return

        # create a saving directory
        save_path = './data/bert_data'
        if os.path.exists('./data/bert_data'):
            cnt = 1
            while True:
                save_path = os.path.join('.', 'data', 'bert_data_' + str(cnt))
                if os.path.exists(save_path):
                    cnt += 1
                else:
                    break
        os.makedirs(save_path, exist_ok=False)

        # embed the raw data with BERT
        print('[INFO] embed the raw data with BERT')
        num_datasets = len(self.raw_data)
        bert_data = list()
        bert_model = load_bert_model()
        for i in range(num_datasets):
            print('### PROCESSING {} out of {}'.format(i+1, num_datasets))
            itr_vec = generate_vecs_bert(bert_model, self.raw_data[i], type='matrix')
            bert_data.append(itr_vec)

            # save the embedded data in the pickle format
            print('[INFO] save the BERT-embedded data into {}'.format(save_path))
            user_name = os.path.basename(self.file_paths[i]).split('_')[0]
            with open(os.path.join(save_path, user_name + '_bert.txt'), 'wb') as f:
                pickle.dump(itr_vec, f)


if __name__ == '__main__':
    # d = DataProcessor(glob('./data/collections_csv/*.csv'))
    # d.csv_loader()

    d = DataProcessor(glob('./data/twitter/*.txt'))
    d.pickle_loader()
    d.bert()
