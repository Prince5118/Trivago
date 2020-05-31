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

train_df = pd.read_csv('processed/train.csv')
test_df = pd.read_csv('processed/test.csv')
item_df = pd.read_csv('processed/item_metadata.csv')
#submission_df = pd.read_csv('processed/submission_popular.csv')

#batch_size = 1024
batch_size = 1024
num_epochs = 5
learning_rate = 5e-3
criterion = nn.L1Loss()


###### functions
"""
Add item features from dictionaries
"""

def get_features(df, item_dict):
    impressions_idx = df[~df['impressions'].isna()].index
    features_lst = []
    prev_idx = -1
    for idx in tqdm(impressions_idx):
        impressions = df['impressions'][idx].split("|")
        tmp_features = []
        for impression in impressions:
            try:
                tmp_features.append(str(item_dict[int(impression)]))
            except:
                tmp_features.append(str(0))
        tmp_features = "|".join(tmp_features)
        tmp_features_lst = [np.nan]*((idx-1)-prev_idx) + [tmp_features]
        features_lst += tmp_features_lst
        prev_idx = idx
    return features_lst

"""
Add 'resident time' feature
"""
def add_resident_time(df):
    timestamp_before_lst = np.array(df['timestamp'])[:-1]
    timestamp_after_lst = np.array(df['timestamp'])[1:]
    
    resident_time_lst = timestamp_after_lst-timestamp_before_lst
    resident_time_lst = np.append(resident_time_lst, [-9999])
    
    step1_idx = np.array(df[df['step']==1].index)
    final_idx = step1_idx-1
    final_idx = np.delete(final_idx, 0)
    final_idx = np.append(final_idx, df.index[-1])
    
    resident_time_lst[final_idx] = -9999
    return resident_time_lst

def copy_features(df):
    impressions_idx = df[~df['impressions'].isna()].index
    impressions_lst = []
    prices_lst = []
    ratings_lst = []
    prev_idx = -1
    for idx in tqdm(impressions_idx):
        impressions = df['impressions'][idx]
        tmp_impressions = [impressions] * (idx-prev_idx)
        impressions_lst += tmp_impressions
        
        prices = df['prices'][idx]
        tmp_prices = [prices] * (idx-prev_idx)
        prices_lst += tmp_prices
        
        ratings = df['ratings'][idx]
        tmp_ratings = [ratings] * (idx-prev_idx)
        ratings_lst += tmp_ratings
        
        prev_idx = idx
    return impressions_lst, prices_lst, ratings_lst

def calculate_diff(df):
    diff_price_lst = []
    diff_rating_lst = []
    ref_price = {}
    ref_rating = {}
    for i in tqdm(range(len(df))):
        ref = df['reference'][i]
        impressions = df['impressions'][i].split("|")
        prices = df['prices'][i].split("|")
        ratings = df['ratings'][i].split("|")
        resident_time = df['resident_time'][i]
        if resident_time != -9999:
            diff_price_lst.append(np.nan)
            diff_rating_lst.append(np.nan)
            if ref in impressions:
                ref_idx = impressions.index(ref)
                ref_price[ref] = int(prices[ref_idx])
                ref_rating[ref] = int(ratings[ref_idx])
        else:
            if len(ref_price) > 0:
                avg_price = sum(ref_price.values())/len(ref_price)
                min_rating = min(ref_rating.values())
                diff_prices = '|'.join([str(int(p)-avg_price) for p in prices])
                diff_ratings = '|'.join([str(1) if int(r)>=min_rating else str(0) for r in ratings])
            else:
                diff_prices = '|'.join([str(0)]*len(prices))
                diff_ratings = '|'.join([str(0)]*len(ratings))
            diff_price_lst.append(diff_prices)
            diff_rating_lst.append(diff_ratings)
            ref_price = {}
            ref_rating = {}
    return diff_price_lst, diff_rating_lst

def filter_diff_impressions(df):
    selected_idx = []
    prev_impressions = ""
    for i, row in tqdm(df.iterrows()):
        step = row['step']
        action_type = row['action_type']
        curr_impressions = row['impressions']
        resident_time = row['resident_time']

        if step == 1:
            first_idx = i
        
        if action_type in ['change of sort order', 'filter selection', 'search for item', 'search for destination', 'search for poi']:
            first_idx = i
        
        if prev_impressions != curr_impressions:
            first_idx = i
        prev_impressions = curr_impressions
        
        if resident_time == -9999:
            last_idx = i
            selected_idx += list(range(first_idx, last_idx+1))
            
    return selected_idx

def get_diff_prices(df):
    diff_city_mean_lst = []
    diff_city_median_lst = []
    diff_nation_mean_lst = []
    diff_nation_median_lst = []
    for i, row in tqdm(df.iterrows()):
        if row['resident_time'] == -9999:
            city = row['city']
            nation = row['nation']
            prices = row['prices'].split("|")
            try:
                city_mean = city_price_dict[city][0]
                city_median = city_price_dict[city][1]
                nation_mean = nation_price_dict[nation][0]
                nation_median = nation_price_dict[nation][1]
                diff_city_mean_lst.append('|'.join([str(int(p)-city_mean) for p in prices]))
                diff_city_median_lst.append('|'.join([str(int(p)-city_median) for p in prices]))
                diff_nation_mean_lst.append('|'.join([str(int(p)-nation_mean) for p in prices]))
                diff_nation_median_lst.append('|'.join([str(int(p)-nation_median) for p in prices]))
            except:
                diff_city_mean_lst.append('|'.join([str(0)]*len(prices)))
                diff_city_median_lst.append('|'.join([str(0)]*len(prices)))
                diff_nation_mean_lst.append('|'.join([str(0)]*len(prices)))
                diff_nation_median_lst.append('|'.join([str(0)]*len(prices)))
        else:
            diff_city_mean_lst.append(np.nan)
            diff_city_median_lst.append(np.nan)
            diff_nation_mean_lst.append(np.nan)
            diff_nation_median_lst.append(np.nan)
    return diff_city_mean_lst, diff_city_median_lst, diff_nation_mean_lst, diff_nation_median_lst

##### end of functions#######



if __name__ == '__main__':
	impressions_idx = train_df[~train_df['impressions'].isna()].index
	clicked_items = list(set([int(item_id) for item_id in list(train_df.loc[impressions_idx]['reference'])]))


	clicked_item_properties = []
	for item in tqdm(clicked_items):
	    try:
	        clicked_item_properties += list(item_df[item_df['item_id']==item]['properties'])[0].split("|")
	    except:
	        pass
	clicked_item_property_count = Counter(clicked_item_properties)

	for k, v in clicked_item_property_count.items():
	    if v > 150000:
	        print (k, v)

	item_hotel = {}
	item_star = {}
	item_tv = {}
	item_shower = {}
	item_window = {}
	item_car = {}
	item_wifi = {}
	item_nosmoke = {}
	for i in tqdm(range(len(item_df))):
	    item_id = item_df['item_id'][i]
	    properties = item_df['properties'][i].split("|")
	    stars = [p for p in properties if 'Star' in p and len(p)==6]
	    if 'Hotel' in properties:
	        item_hotel[item_id] = 1
	        if len(stars) > 0:
	            item_star[item_id] = int(stars[0][0])
	        else:
	            item_star[item_id] = 0
	    else:
	        item_hotel[item_id] = 0
	        item_star[item_id] = 0
	    
	    item_tv[item_id] = 1 if 'Television' in properties else 0
	    item_shower[item_id] = 1 if 'Shower' in properties else 0
	    item_window[item_id] = 1 if 'Openable Windows' in properties else 0
	    item_car[item_id] = 1 if 'Car Park' in properties else 0
	    item_wifi[item_id] = 1 if 'WiFi (Public Areas)' in properties else 0
	    item_wifi[item_id] = 1 if 'WiFi (Rooms)' in properties else item_wifi[item_id]
	    item_nosmoke[item_id] = 1 if 'Non-Smoking Rooms' in properties else 0



	#########################
	#######################
	print('Here - rating dictionary using item metadata')

	properties = []
	for i in range(len(item_df)):
	    properties += item_df['properties'][i].split("|")
	properties = list(set(properties))
	ratings = [p for p in properties if 'Rating' in p]
	print (ratings)

	item_rating = {}
	for i in range(len(item_df)):
	    item_id = item_df['item_id'][i]
	    properties = item_df['properties'][i].split("|")
	    rating = [p for p in properties if p in ratings]
	    item_rating[item_id] = len(rating)+1

	user_lst = list(set(list(set(train_df['user_id']))+list(set(test_df['user_id']))))
	user_items = {}
	for user in user_lst:
	    user_items[user] = []

	viewed_lst = []
	for i, row in tqdm(train_df.iterrows()):
	    u_id = row['user_id']
	    action = row['action_type']
	    ref = row['reference']
	    impressions = row['impressions']
	    
	    if action == "clickout item":
	        viewed = "|".join([str(1) if int(item) in user_items[u_id] else str(0) for item in impressions.split("|")])
	        viewed_lst.append(viewed)
	    else:
	        viewed_lst.append(np.nan)
	        
	    try:
	        if int(ref) not in user_items[u_id]:
	            user_items[u_id].append(int(ref))
	    except:
	        pass

	train_df['viewed'] = viewed_lst

	viewed_lst = []
	for i, row in tqdm(test_df.iterrows()):
	    u_id = row['user_id']
	    action = row['action_type']
	    ref = row['reference']
	    impressions = row['impressions']
	    
	    if action == "clickout item":
	        viewed = "|".join([str(1) if int(item) in user_items[u_id] else str(0) for item in impressions.split("|")])
	        viewed_lst.append(viewed)
	    else:
	        viewed_lst.append(np.nan)
	        
	    try:
	        if int(ref) not in user_items[u_id]:
	            user_items[u_id].append(int(ref))
	    except:
	        pass
	test_df['viewed'] = viewed_lst


	"""
	Delete meaningless train dataset for us¶
	delete sessions that have no 'clickout' action and delete the steps after the last 'clickout' action within a session.
	"""
	print('Here - Delete meaningless train dataset for us')

	act_lst = []
	selected_idx = []

	for i in tqdm(range(len(train_df))):
	    step = train_df['step'][i]
	    if i != 0 and step == 1:
	        first_idx = i-(len(act_lst))
	        clickout_idx = np.where(np.array(act_lst)=='clickout item')[0]
	        if len(clickout_idx) != 0:
	            last_idx = first_idx + clickout_idx[-1]
	            selected_idx += list(range(first_idx, last_idx+1))
	        else:
	            last_idx = -9999
	        act_lst = []
	    act = train_df['action_type'][i]
	    act_lst.append(act)
	    
	selected_idx += list(range(15932973, 15932992))

	new_train_df = pd.DataFrame(train_df.loc()[selected_idx], columns=train_df.columns)
	new_train_df = new_train_df.reset_index(drop=True)

	selected_idx = []
	for i in tqdm(range(len(test_df))):
	    step = test_df['step'][i]
	    action_type = test_df['action_type'][i]
	    ref = test_df['reference'][i]
	    if step == 1:
	        first_idx = i
	    if action_type == 'clickout item' and ref is np.nan:
	        last_idx = i
	        selected_idx += list(range(first_idx, last_idx+1))


	new_test_df = pd.DataFrame(test_df.loc[selected_idx], columns=test_df.columns)
	new_test_df = new_test_df.reset_index(drop=True)

	"""
	Add item features from dictionaries
	"""



	print('Add item features from dictionaries')
	names = ['ratings', 'hotel', 'star', 'tv', 'shower', 'window', 'car', 'wifi', 'nosmoke']
	features_dict = [item_rating, item_hotel, item_star, item_tv, item_shower, item_window, item_car, item_wifi, item_nosmoke]

	for name, feature_dict in list(zip(names, features_dict)):
	    new_train_df[name] = get_features(new_train_df, feature_dict)
	    new_test_df[name] = get_features(new_test_df, feature_dict)

	#Add 'resident time' feature

	new_train_df['resident_time'] = add_resident_time(new_train_df)
	new_test_df['resident_time'] = add_resident_time(new_test_df)

	"""
	Add 'price difference' feature and 'rating difference' feature
	'price difference' is the difference between the accommodations on the screen and the price that the user has seen within the same session
	'rating difference' is the binary feature. The value is 1 if the rating of the accommodations on the screen is equal to or greater than the minimum rating that the user has seen within the same session, otherwise 0.
	If user did not see any item, these features are filled with 0.
	"""


	new_train_df['impressions'], new_train_df['prices'], new_train_df['ratings'] = copy_features(new_train_df)
	new_test_df['impressions'], new_test_df['prices'], new_test_df['ratings'] = copy_features(new_test_df)



	new_train_df['diff_prices'], new_train_df['diff_ratings'] = calculate_diff(new_train_df)
	new_test_df['diff_prices'], new_test_df['diff_ratings'] = calculate_diff(new_test_df)



	selected_idx = filter_diff_impressions(new_train_df)
	filtered_train_df = pd.DataFrame(new_train_df.loc[selected_idx], columns=new_train_df.columns)
	filtered_train_df = filtered_train_df.reset_index(drop=True)

	selected_idx = filter_diff_impressions(new_test_df)
	filtered_test_df = pd.DataFrame(new_test_df.loc[selected_idx], columns=new_test_df.columns)
	filtered_test_df = filtered_test_df.reset_index(drop=True)

	"""
	Use the average price per nation or city
	"""

	nation_lst = []
	for i in tqdm(range(len(filtered_train_df))):
	    nation_lst.append(filtered_train_df['city'][i].split(", ")[1])
	filtered_train_df['nation'] = nation_lst

	test_nation_lst = []
	for i in tqdm(range(len(filtered_test_df))):
	    test_nation_lst.append(filtered_test_df['city'][i].split(", ")[1])
	filtered_test_df['nation'] = test_nation_lst

	nation_price_dict = {}
	nations = list(set(nation_lst))
	last_step = filtered_train_df[filtered_train_df['resident_time']==-9999]
	for nation in tqdm(nations):
	    tmp = last_step[last_step['nation']==nation]

	    prices_all = []
	    for i in range(len(tmp)):
	        prices = [int(p) for p in tmp['prices'].iloc[i].split("|")]
	        prices_all += prices
	    nation_price_dict[nation] = [np.mean(prices_all), np.median(prices_all)]

	city_price_dict = {}
	cities = list(set(filtered_train_df['city']))
	last_step = filtered_train_df[filtered_train_df['resident_time']==-9999]
	for city in tqdm(cities):
	    tmp = last_step[last_step['city']==city]
	    
	    prices_all = []
	    for i in range(len(tmp)):
	        prices = [int(p) for p in tmp['prices'].iloc[i].split("|")]
	        prices_all += prices
	    city_price_dict[city] = [np.mean(prices_all), np.median(prices_all)]




	print('final step - saving files')

	filtered_train_df['diff_city_mean'], filtered_train_df['diff_city_median'], filtered_train_df['diff_nation_mean'], filtered_train_df['diff_nation_median'] = get_diff_prices(filtered_train_df)
	filtered_test_df['diff_city_mean'], filtered_test_df['diff_city_median'], filtered_test_df['diff_nation_mean'], filtered_test_df['diff_nation_median'] = get_diff_prices(filtered_test_df)

	filtered_train_df.to_csv('processed/train_final.csv', index=False)
	filtered_test_df.to_csv('processed/test_final.csv', index=False)