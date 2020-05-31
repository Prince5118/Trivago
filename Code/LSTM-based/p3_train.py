#Feature nomalization and extension (Train)
import pickle
import pandas as pd
import numpy as np


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
for i in range(len(sessions)):
    sessions[i] = sessions[i][35:]

# For reciprocal price,
price = np.array(displays)[:,1,:].tolist()

# display:
# idx(0), price(1), hotel(2), star(3), tv(4), shower(5), window(6), car(7), wifi(8), nosmoke(9), ratings(10)
# viewed(11), diff_prices(12), diff_ratings(13)
# diff_city_mean(14), diff_city_median(15), diff_nation_mean(16), diff_nation_median(17)

# Normalize
print('Starting Normalization.....')
PRICE_THRESHOLD = 1000
for i in range(len(displays)):
    displays[i][0] = [1/p if p!=0 else 0 for p in displays[i][0]]
    displays[i][1] = [p/PRICE_THRESHOLD if p<PRICE_THRESHOLD else 1 for p in displays[i][1]]
    displays[i][3] = [r/5 for r in displays[i][3]]
    displays[i][10] = [r/5 for r in displays[i][10]]
    
    max_diff = max([max(displays[i][12]),-min(displays[i][12])])
    if max_diff != 0:
        displays[i][12] = [p/max_diff for p in displays[i][12]]
        
    displays[i][14] = [p/PRICE_THRESHOLD if np.abs(p)<PRICE_THRESHOLD else p/np.abs(p) for p in displays[i][14]]
    displays[i][15] = [p/PRICE_THRESHOLD if np.abs(p)<PRICE_THRESHOLD else p/np.abs(p) for p in displays[i][15]]
    displays[i][16] = [p/PRICE_THRESHOLD if np.abs(p)<PRICE_THRESHOLD else p/np.abs(p) for p in displays[i][16]]
    displays[i][17] = [p/PRICE_THRESHOLD if np.abs(p)<PRICE_THRESHOLD else p/np.abs(p) for p in displays[i][17]]
print('Ending Normalization.....')

# Feature Extension
# idx_sqrt(18), idx_square(19), price_sqrt(20), price_square(21)
# star_sqrt(22) start_square(23), rating_sqrt(24), rating_square(25)
# price_diff_sqrt(26), price_diff_square(27), price_mean_on_display(28), price_median_on_display(29)
# diff_city_mean_sqrt(30), diff_city_mean_square(31), diff_city_median_sqrt(32), diff_city_median_square(33)
# diff_nation_mean_sqrt(34), diff_nation_mean_square(35), diff_nation_median_sqrt(36), diff_nation_median_square(37)
# reciprocal_price(38)
print('Starting Feature extension.....')

for i in range(len(displays)):
    idx_sqrt = np.sqrt(displays[i][0]).tolist()
    idx_square = np.square(displays[i][0]).tolist()
    price_sqrt = np.sqrt(displays[i][1]).tolist()
    price_square = np.square(displays[i][1]).tolist()
    star_sqrt = np.sqrt(displays[i][3]).tolist()
    star_square = np.square(displays[i][3]).tolist()
    rating_sqrt = np.sqrt(displays[i][10]).tolist()
    rating_square = np.square(displays[i][10]).tolist()

    price_diff_sqrt = (np.where(np.array(displays[i][12])>=0,1,-1)*np.sqrt(np.abs(displays[i][12]))).tolist()
    price_diff_square = (np.where(np.array(displays[i][12])>=0,1,-1)*np.square(displays[i][12])).tolist()
    price_mean_on_display = (np.array(displays[i][1])-np.mean(displays[i][1])).tolist()
    price_median_on_display = (np.array(displays[i][1])-np.median(displays[i][1])).tolist()

    diff_city_mean_sqrt = (np.where(np.array(displays[i][14])>=0,1,-1)*np.sqrt(np.abs(displays[i][14]))).tolist()
    diff_city_mean_square = (np.where(np.array(displays[i][14])>=0,1,-1)*np.square(displays[i][14])).tolist()
    diff_city_median_sqrt = (np.where(np.array(displays[i][15])>=0,1,-1)*np.sqrt(np.abs(displays[i][15]))).tolist()
    diff_city_median_square = (np.where(np.array(displays[i][15])>=0,1,-1)*np.square(displays[i][15])).tolist()
    diff_nation_mean_sqrt = (np.where(np.array(displays[i][16])>=0,1,-1)*np.sqrt(np.abs(displays[i][16]))).tolist()
    diff_nation_mean_square = (np.where(np.array(displays[i][16])>=0,1,-1)*np.square(displays[i][16])).tolist()
    diff_nation_median_sqrt = (np.where(np.array(displays[i][17])>=0,1,-1)*np.sqrt(np.abs(displays[i][17]))).tolist()
    diff_nation_median_square = (np.where(np.array(displays[i][17])>=0,1,-1)*np.square(displays[i][17])).tolist()

    reciprocal_price = [1/p if p!=0 else 0 for p in price[i]]

    displays[i].extend([idx_sqrt, idx_square, price_sqrt, price_square, star_sqrt, star_square, rating_sqrt, rating_square])
    displays[i].extend([price_diff_sqrt, price_diff_square, price_mean_on_display, price_median_on_display])
    displays[i].extend([diff_city_mean_sqrt, diff_city_mean_square, diff_city_median_sqrt, diff_city_median_square, diff_nation_mean_sqrt, diff_nation_mean_square, diff_nation_median_sqrt, diff_nation_median_square, reciprocal_price])
print('Endnig Feature Extension')


train_dataset = list(zip(np.array(sessions), np.array(displays), np.array(encodings), np.array(criteria), np.array(devices), np.array(clicked_item)))
print('Starting to Dump File..')
with open("data/train_dataset.pickle", "wb") as f:
    pickle.dump(train_dataset, f)
#if __name__ == '__main__':

