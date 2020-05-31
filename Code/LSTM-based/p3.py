#Feature nomalization and extension (Train)
import csv
import copy
import pickle
import pprint
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data.dataset import random_split

from collections import Counter
from tqdm import tqdm


train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
item_df = pd.read_csv('data/item_metadata.csv')
submission_df = pd.read_csv('data/submission_popular.csv')


with open('data/train_final.pickle', 'rb') as f:
    sessions = pickle.load(f)
with open('data/train_display_final.pickle', 'rb') as f:
    displays = pickle.load(f)
with open('data/train_encoding_final.pickle', 'rb') as f:
    encodings = pickle.load(f)
with open('data/train_criteria_final.pickle', 'rb') as f:
    criteria = pickle.load(f)
with open('data/train_device_final.pickle', 'rb') as f:
    devices = pickle.load(f)
with open('data/label_final.pickle', 'rb') as f:
    clicked_item = pickle.load(f)


# set max length as 15
for i in range(len(test_sessions)):
    test_sessions[i] = test_sessions[i][35:]

# For reciprocal price,
test_price = np.array(test_displays)[:,1,:].tolist()

# display:
# idx(0), price(1), hotel(2), star(3), tv(4), shower(5), window(6), car(7), wifi(8), nosmoke(9), ratings(10)
# viewed(11), diff_prices(12), diff_ratings(13)
# diff_city_mean(14), diff_city_median(15), diff_nation_mean(16), diff_nation_median(17)

# Normalize
PRICE_THRESHOLD = 1000
for i in range(len(test_displays)):
    test_displays[i][0] = [1/p if p!=0 else 0 for p in test_displays[i][0]]
    test_displays[i][1] = [p/PRICE_THRESHOLD if p<PRICE_THRESHOLD else 1 for p in test_displays[i][1]]
    test_displays[i][3] = [r/5 for r in test_displays[i][3]]
    test_displays[i][10] = [r/5 for r in test_displays[i][10]]
    
    max_diff = max([max(test_displays[i][12]),-min(test_displays[i][12])])
    if max_diff != 0:
        test_displays[i][12] = [p/max_diff for p in test_displays[i][12]]
        
    test_displays[i][14] = [p/PRICE_THRESHOLD if np.abs(p)<PRICE_THRESHOLD else p/np.abs(p) for p in test_displays[i][14]]
    test_displays[i][15] = [p/PRICE_THRESHOLD if np.abs(p)<PRICE_THRESHOLD else p/np.abs(p) for p in test_displays[i][15]]
    test_displays[i][16] = [p/PRICE_THRESHOLD if np.abs(p)<PRICE_THRESHOLD else p/np.abs(p) for p in test_displays[i][16]]
    test_displays[i][17] = [p/PRICE_THRESHOLD if np.abs(p)<PRICE_THRESHOLD else p/np.abs(p) for p in test_displays[i][17]]
    
test_displays = np.nan_to_num(test_displays).tolist()

# Feature Extension
# idx_sqrt(18), idx_square(19), price_sqrt(20), price_square(21)
# star_sqrt(22) start_square(23), rating_sqrt(24), rating_square(25)
# price_diff_sqrt(26), price_diff_square(27), price_mean_on_display(28), price_median_on_display(29)
# diff_city_mean_sqrt(30), diff_city_mean_square(31), diff_city_median_sqrt(32), diff_city_median_square(33)
# diff_nation_mean_sqrt(34), diff_nation_mean_square(35), diff_nation_median_sqrt(36), diff_nation_median_square(37)
# reciprocal_price(38)
for i in range(len(test_displays)):
    idx_sqrt = np.sqrt(test_displays[i][0]).tolist()
    idx_square = np.square(test_displays[i][0]).tolist()
    price_sqrt = np.sqrt(test_displays[i][1]).tolist()
    price_square = np.square(test_displays[i][1]).tolist()
    star_sqrt = np.sqrt(test_displays[i][3]).tolist()
    star_square = np.square(test_displays[i][3]).tolist()
    rating_sqrt = np.sqrt(test_displays[i][10]).tolist()
    rating_square = np.square(test_displays[i][10]).tolist()

    price_diff_sqrt = (np.where(np.array(test_displays[i][12])>=0,1,-1)*np.sqrt(np.abs(test_displays[i][12]))).tolist()
    price_diff_square = (np.where(np.array(test_displays[i][12])>=0,1,-1)*np.square(test_displays[i][12])).tolist()
    price_mean_on_display = (np.array(test_displays[i][1])-np.mean(test_displays[i][1])).tolist()
    price_median_on_display = (np.array(test_displays[i][1])-np.median(test_displays[i][1])).tolist()

    diff_city_mean_sqrt = (np.where(np.array(test_displays[i][14])>=0,1,-1)*np.sqrt(np.abs(test_displays[i][14]))).tolist()
    diff_city_mean_square = (np.where(np.array(test_displays[i][14])>=0,1,-1)*np.square(test_displays[i][14])).tolist()
    diff_city_median_sqrt = (np.where(np.array(test_displays[i][15])>=0,1,-1)*np.sqrt(np.abs(test_displays[i][15]))).tolist()
    diff_city_median_square = (np.where(np.array(test_displays[i][15])>=0,1,-1)*np.square(test_displays[i][15])).tolist()
    diff_nation_mean_sqrt = (np.where(np.array(test_displays[i][16])>=0,1,-1)*np.sqrt(np.abs(test_displays[i][16]))).tolist()
    diff_nation_mean_square = (np.where(np.array(test_displays[i][16])>=0,1,-1)*np.square(test_displays[i][16])).tolist()
    diff_nation_median_sqrt = (np.where(np.array(test_displays[i][17])>=0,1,-1)*np.sqrt(np.abs(test_displays[i][17]))).tolist()
    diff_nation_median_square = (np.where(np.array(test_displays[i][17])>=0,1,-1)*np.square(test_displays[i][17])).tolist()

    reciprocal_price = [1/p if p!=0 else 0 for p in test_price[i]]

    test_displays[i].extend([idx_sqrt, idx_square, price_sqrt, price_square, star_sqrt, star_square, rating_sqrt, rating_square])
    test_displays[i].extend([price_diff_sqrt, price_diff_square, price_mean_on_display, price_median_on_display])
    test_displays[i].extend([diff_city_mean_sqrt, diff_city_mean_square, diff_city_median_sqrt, diff_city_median_square, diff_nation_mean_sqrt, diff_nation_mean_square, diff_nation_median_sqrt, diff_nation_median_square, reciprocal_price])


test_dataset = list(zip(np.array(test_sessions), np.array(test_displays), np.array(test_encodings), np.array(test_criteria), np.array(test_devices)))

with open("data/test_dataset.pickle", "wb") as f:
    pickle.dump(test_dataset, f)