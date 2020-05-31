#From dataframe to real using data
import csv
import pickle
import pandas as pd
import numpy as np
from collections import Counter



def create_train_data():
    EMBEDDING_SIZE = 10+25+1
    THRESHOLD_TIME = 30.0
    print('Opening train_final')
    with open('processed/train_final.csv', encoding="utf-8") as train_f:
        rdr = csv.reader(train_f)
        next(rdr)

        sequence_lst = []
        train_lst = []
        display_lst = []
        encoding_lst = []
        device_lst = []
        criteria_lst = []
        label_lst = []

        for idx, line in enumerate(rdr):
            u_id = line[0]
            s_id = line[1]
            timestamp = line[2]
            step = line[3]
            action_type = line[4]
            reference = line[5]
            platform = line[6]
            city = line[7]
            device = line[8].split("|")
            current_filters = line[9]
            impressions = line[10].split("|")
            prices = line[11].split("|")
            viewed = line[12].split("|")
            ratings = line[13].split("|")
            hotel = line[14].split("|")
            star = line[15].split("|")
            tv = line[16].split("|")
            shower = line[17].split("|")
            window = line[18].split("|")
            car = line[19].split("|")
            wifi = line[20].split("|")
            nosmoke = line[21].split("|")
            resident_time = line[22]
            diff_prices = line[23].split("|")
            diff_ratings = line[24].split("|")
            nation = line[25]
            diff_city_mean = line[26].split("|")
            diff_city_median = line[27].split("|")
            diff_nation_mean = line[28].split("|")
            diff_nation_median = line[29].split("|")
            
            action_embedding = action_type_one_hot[action_type]
            ref_id_embedding = [0]*25
            if reference in impressions:
                ref_idx = impressions.index(reference)
                ref_id_embedding[ref_idx] += 1.

            time_embedding = [min([int(resident_time)+1, THRESHOLD_TIME])/THRESHOLD_TIME]
            step_embedding = action_embedding + ref_id_embedding + time_embedding
            
            if action_type in numeric_action_type:
                if reference in impressions:
                    sequence_lst.append(step_embedding)
            else:
                sequence_lst.append(step_embedding)
            
            if resident_time == "-9999" and len(sequence_lst)>0:
                sequence_lst.pop()
                if reference in impressions:
                    sequence_lst = sequence_lst[-50:]
                    sequence_lst = [([0] * EMBEDDING_SIZE) for _ in range((50 - len(sequence_lst)))] + sequence_lst
                    train_lst.append(sequence_lst)
                    
                    idx_lst = list(range(1,len(impressions)+1))
                    idx_lst += [0] * (25-len(idx_lst))
                    prices = [float(p) for p in prices]
                    prices += [0] * (25-len(prices))
                    viewed = [int(x) for x in viewed]
                    viewed += [0] * (25-len(viewed))
                    ratings = [int(x) for x in ratings]
                    ratings += [0] * (25-len(ratings))
                    hotel = [int(x) for x in hotel]
                    hotel += [0] * (25-len(hotel))
                    star = [int(x) for x in star]
                    star += [0] * (25-len(star))
                    tv = [int(x) for x in tv]
                    tv += [0] * (25-len(tv))
                    shower = [int(x) for x in shower]
                    shower += [0] * (25-len(shower))
                    window = [int(x) for x in window]
                    window += [0] * (25-len(window))
                    car = [int(x) for x in car]
                    car += [0] * (25-len(car))
                    wifi = [int(x) for x in wifi]
                    wifi += [0] * (25-len(wifi))
                    nosmoke = [int(x) for x in nosmoke]
                    nosmoke += [0] * (25-len(nosmoke))                    
                    
                    diff_prices = [float(x) for x in diff_prices]
                    diff_prices += [0] * (25-len(diff_prices))
                    diff_ratings = [float(x) for x in diff_ratings]
                    diff_ratings += [0] * (25-len(diff_ratings))
                    diff_city_mean = [float(x) for x in diff_city_mean]
                    diff_city_mean += [0] * (25-len(diff_city_mean))
                    diff_city_median = [float(x) for x in diff_city_median]
                    diff_city_median += [0] * (25-len(diff_city_median))
                    diff_nation_mean = [float(x) for x in diff_nation_mean]
                    diff_nation_mean += [0] * (25-len(diff_nation_mean))
                    diff_nation_median = [float(x) for x in diff_nation_median]
                    diff_nation_median += [0] * (25-len(diff_nation_median))
                    
                    features = [idx_lst, prices, hotel, star, tv, shower, window, car, wifi, nosmoke, ratings, viewed, diff_prices, diff_ratings, diff_city_mean, diff_city_median, diff_nation_mean, diff_nation_median]
                    display_lst.append(features)
                    
                    item_encoding = []
                    for j in range(25):
                        try:
                            item_encoding.append(item_encoding_dict[int(impressions[j])].tolist())
                        except:
                            item_encoding.append([0.]*32)
                    item_encoding = np.transpose(np.array(item_encoding), (1,0)).tolist()
                    encoding_lst.append(item_encoding)
                    
                    filtered_criteria = [0]*12
                    try:
                        current_filters = current_filters.split("|")
                        for i in range(len(common_filters)):
                            for fil in current_filters:
                                if common_filters[i] in fil:
                                    filtered_criteria[i] = 1
                        criteria_lst.append(filtered_criteria)
                    except:
                        criteria_lst.append(filtered_criteria)
                    
                    if device == "desktop":
                        device_lst.append([1,0,0])
                    elif device == "mobile":
                        device_lst.append([0,1,0])
                    else:
                        device_lst.append([0,0,1])
                    
                    label_lst.append(impressions.index(reference))
                sequence_lst = []
        print('Starting Dumping Train files')
        with open('processed/train_final.pickle', 'wb') as f:
            pickle.dump(train_lst, f)
        with open('processed/train_display_final.pickle', 'wb') as f:
            pickle.dump(display_lst, f)
        with open('processed/train_encoding_final.pickle', 'wb') as f:
            pickle.dump(encoding_lst, f)
        print('Midway dumping train files')
        with open('processed/train_criteria_final.pickle', 'wb') as f:
            pickle.dump(criteria_lst, f)
        with open('processed/train_device_final.pickle', 'wb') as f:
            pickle.dump(device_lst, f)
        with open('processed/label_final.pickle', 'wb') as f:
            pickle.dump(label_lst, f)

def create_test_data():
    EMBEDDING_SIZE = 10+25+1
    THRESHOLD_TIME = 30.0
    print('Opening test_final')
    with open('processed/test_final.csv', encoding="utf-8") as test_f:
        rdr = csv.reader(test_f)
        next(rdr)

        sequence_lst = []
        test_lst = []
        display_lst = []
        encoding_lst = []
        criteria_lst = []
        device_lst = []
        line_lst = []

        for idx, line in enumerate(rdr):
            u_id = line[0]
            s_id = line[1]
            timestamp = line[2]
            step = line[3]
            action_type = line[4]
            reference = line[5]
            platform = line[6]
            city = line[7]
            device = line[8].split("|")
            current_filters = line[9]
            impressions = line[10].split("|")
            prices = line[11].split("|")
            viewed = line[12].split("|")
            ratings = line[13].split("|")
            hotel = line[14].split("|")
            star = line[15].split("|")
            tv = line[16].split("|")
            shower = line[17].split("|")
            window = line[18].split("|")
            car = line[19].split("|")
            wifi = line[20].split("|")
            nosmoke = line[21].split("|")
            resident_time = line[22]
            diff_prices = line[23].split("|")
            diff_ratings = line[24].split("|")
            nation = line[25]
            diff_city_mean = line[26].split("|")
            diff_city_median = line[27].split("|")
            diff_nation_mean = line[28].split("|")
            diff_nation_median = line[29].split("|")
            
            action_embedding = action_type_one_hot[action_type]
            ref_id_embedding = [0]*25
            if reference in impressions:
                ref_idx = impressions.index(reference)
                ref_id_embedding[ref_idx] += 1.

            time_embedding = [min([int(resident_time)+1, THRESHOLD_TIME])/THRESHOLD_TIME]
            step_embedding = action_embedding + ref_id_embedding + time_embedding
            
            if action_type in numeric_action_type:
                if reference in impressions:
                    sequence_lst.append(step_embedding)
            else:
                sequence_lst.append(step_embedding)
            
            if resident_time == "-9999":
                sequence_lst = sequence_lst[-50:]
                sequence_lst = [([0] * EMBEDDING_SIZE) for _ in range((50 - len(sequence_lst)))] + sequence_lst
                test_lst.append(sequence_lst)    
            
                idx_lst = list(range(1,len(impressions)+1))
                idx_lst += [0] * (25-len(idx_lst))
                prices = [float(p) for p in prices]
                prices += [0] * (25-len(prices))
                viewed = [int(x) for x in viewed]
                viewed += [0] * (25-len(viewed))
                ratings = [int(x) for x in ratings]
                ratings += [0] * (25-len(ratings))
                hotel = [int(x) for x in hotel]
                hotel += [0] * (25-len(hotel))
                star = [int(x) for x in star]
                star += [0] * (25-len(star))
                tv = [int(x) for x in tv]
                tv += [0] * (25-len(tv))
                shower = [int(x) for x in shower]
                shower += [0] * (25-len(shower))
                window = [int(x) for x in window]
                window += [0] * (25-len(window))
                car = [int(x) for x in car]
                car += [0] * (25-len(car))
                wifi = [int(x) for x in wifi]
                wifi += [0] * (25-len(wifi))
                nosmoke = [int(x) for x in nosmoke]
                nosmoke += [0] * (25-len(nosmoke))
                
                diff_prices = [float(p) for p in diff_prices]
                diff_prices += [0] * (25-len(diff_prices))
                diff_ratings = [float(r) for r in diff_ratings]
                diff_ratings += [0] * (25-len(diff_ratings))
                diff_city_mean = [float(x) for x in diff_city_mean]
                diff_city_mean += [0] * (25-len(diff_city_mean))
                diff_city_median = [float(x) for x in diff_city_median]
                diff_city_median += [0] * (25-len(diff_city_median))
                diff_nation_mean = [float(x) for x in diff_nation_mean]
                diff_nation_mean += [0] * (25-len(diff_nation_mean))
                diff_nation_median = [float(x) for x in diff_nation_median]
                diff_nation_median += [0] * (25-len(diff_nation_median))
                    
                features = [idx_lst, prices, hotel, star, tv, shower, window, car, wifi, nosmoke, ratings, viewed, diff_prices, diff_ratings, diff_city_mean, diff_city_median, diff_nation_mean, diff_nation_median]
                display_lst.append(features)
                
                item_encoding = []
                for j in range(25):
                    try:
                        item_encoding.append(item_encoding_dict[int(impressions[j])].tolist())
                    except:
                        item_encoding.append([0.]*32)
                item_encoding = np.transpose(np.array(item_encoding), (1,0)).tolist()
                encoding_lst.append(item_encoding)
                
                filtered_criteria = [0]*12
                try:
                    current_filters = current_filters.split("|")
                    for i in range(len(common_filters)):
                        for fil in current_filters:
                            if common_filters[i] in fil:
                                filtered_criteria[i] = 1
                    criteria_lst.append(filtered_criteria)
                except:
                    criteria_lst.append(filtered_criteria)
                        
                if device == "desktop":
                    device_lst.append([1,0,0])
                elif device == "mobile":
                    device_lst.append([0,1,0])
                else:
                    device_lst.append([0,0,1])
                                      
                line_lst.append([u_id, s_id, timestamp, step, impressions])
                sequence_lst = []
        print('Starting Dumping test files')
        with open('processed/test_final.pickle', 'wb') as f:
            pickle.dump(test_lst, f)
        with open('processed/test_display_final.pickle', 'wb') as f:
            pickle.dump(display_lst, f)
        with open('processed/test_encoding_final.pickle', 'wb') as f:
            pickle.dump(encoding_lst, f)
        print('Midway Dumpping Test Files')
        with open('processed/test_criteria_final.pickle', 'wb') as f:
            pickle.dump(criteria_lst, f)
        with open('processed/test_device_final.pickle', 'wb') as f:
            pickle.dump(device_lst, f)
        with open('processed/line_final.pickle', 'wb') as f:
            pickle.dump(line_lst, f)
#### end of funcitons
            
if __name__ == '__main__':
    
    print('starting reading trainfinal')
    train_df = pd.read_csv('processed/train_final.csv')
    print('strating reading test final')
    test_df = pd.read_csv('processed/test_final.csv')
    
    with open('processed/item_encoding_dict.pickle', 'rb') as f:
        item_encoding_dict = pickle.load(f)
    print('Done reading item encoding')
    filter_criteria = Counter(train_df[train_df['action_type']=='filter selection']['reference'])
    for k, v in filter_criteria.items():
        if v > 2000:
            print (k, v)
    
    
    common_filters = ['Price', 'Rating', 'Distance', 'Value', 'Hotel', 'Star', 'Hostal', 'Motel', 'Apartment', 'Breakfast', 'WiFi', 'Park']
    action_type_lst = ['clickout item', 'interaction item rating', 'interaction item info', 'interaction item image', 'interaction item deals',
                       'change of sort order', 'filter selection', 'search for item', 'search for destination', 'search for poi']
    numeric_action_type = ['clickout item', 'interaction item rating', 'interaction item info', 'interaction item image', 'interaction item deals', 'search for item']
    
    action_type_one_hot = {}
    for i in range(len(action_type_lst)):
        tmp = [0]*10
        tmp[i] = 1
        action_type_one_hot[action_type_lst[i]] = tmp
    print (action_type_one_hot)
    
    print("... create train data ...")
    create_train_data()
    print("... end train data ...")
    
    
    print("... create test data ...")
    create_test_data()
    print("... end test data ...")