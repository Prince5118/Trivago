from pathlib import Path
import pandas as pd

from  baseline_algorithm import functions as f


pd.set_option('display.max_columns', None)

def main(data_path):

    # calculate path to files
    data_directory = Path(data_path) if data_path else " " #else default_data_directory
    train_csv = data_directory.joinpath('train.csv')
    test_csv = data_directory.joinpath('test.csv')
    subm_csv = data_directory.joinpath('submission_popular.csv')
    df_meta = data_directory.joinpath('5_features.csv')

    print(f"Reading {train_csv} ...")
    df_train = pd.read_csv(train_csv)
    print(f"Reading {test_csv} ...")
    df_test = pd.read_csv(test_csv)
    print(f"Reading {df_meta} ...")
    df_meta = pd.read_csv(df_meta)

    print("Get popular items...")
    df_popular = f.get_popularity_and_information(df_train, df_meta)
    #df_popular = f.get_ctr_and_n_clicks(df_train)
    #Najbardziej popularne hotele na podstawie ctr i n clicks
   # print(df_popular.head())

    print("Identify target rows...")
    df_target = f.get_submission_target(df_test) # -> wybranie tych reference
                                                 # gdzie mamy NaN i click

    print("Get recommendations...")
    df_expl = f.explode(df_target, "impressions") #rozbijamy te impressions z df_target
    #print(df_expl)
    print("Calc recommendation")
    df_out = f.calc_recommendation(df_expl, df_popular) #
    #print(df_out)

    print(f"Writing {subm_csv}...")
    df_out.to_csv(subm_csv, index=False)

    print("Finished calculating recommendations.")

# współczynnik

if __name__ == '__main__':
    main('../../data')
