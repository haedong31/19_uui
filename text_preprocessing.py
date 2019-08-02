from glob import glob
import csv
import demoji
import numpy as np
import os
import pickle
import re


# download emoji codes; only execute once on the first use
# demoji.download_codes()


# load data
file_paths = glob('./data/collections_csv/*.csv')
raw_data = list()
num_files = len(file_paths)
for i in range(num_files):
    print('### PROCESSING {} out of {}'.format(i+1, num_files))

    with open(file_paths[i], 'rt') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # skip the header
        
        itr_list = list()
        for r in csv_reader:
            itr_list.append(r[2])
        
        raw_data.append(itr_list)
raw_data_copy = raw_data

# average number of tweets per user
num_tweets = [len(x) for x in raw_data]
np.average(num_tweets)


# remove emojis and URLs
# demoji.replace(raw_data[0][7], '')
# re.sub(r'https?://\S+', '', raw_data[0][7], flags=re.MULTILINE)
for i in range(num_files):
    print('### PROCESSING {} out of {}'.format(i + 1, num_files))
    for j in range(num_tweets[i]):
        raw_data[i][j] = demoji.replace(raw_data[i][j], '')
        raw_data[i][j] = re.sub(r'https?://\S+', '', raw_data[i][j], flags=re.MULTILINE)

cleaned_data = list()
r = re.compile(r'\s')
for i in range(num_files):
    print('### PROCESSING {} out of {}'.format(i + 1, num_files))
    itr_list = [x for x in raw_data[i] if not r.match(x)]
    itr_list = list(filter(None, itr_list))
    cleaned_data.append(itr_list)

# save the cleaned data
for i in range(num_files):
    print('### PROCESSING {} out of {}'.format(i + 1, num_files))
    save_path = os.path.join('.', 'data', 'twitter', 'user_'+str(i)+'.txt')
    with open(save_path, 'wb') as f:
        pickle.dump(cleaned_data[i], f)

with open(save_path, 'rb') as f:
    test = pickle.load(f)

full = ['webb', 'ellis', '(sportswear)']
regex = re.compile(r'\(.*\)$')
filtered = [i for i in full if not regex.match(i)]