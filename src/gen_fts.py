""" generate new features and save them to data/preprocessed """

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from utils import load_config, load_data
from sklearn.preprocessing import LabelEncoder

# load config and data
config = load_config()
train, test = load_data()

# add new features
def add_features(df):
    df['timestamp'] = pd.to_datetime(df.timestamp)
    df['month'] = df.timestamp.dt.month
    df['hour'] = df.timestamp.dt.hour
    df['dayofweek'] = df.timestamp.dt.dayofweek
    df['dayofmonth'] = df.timestamp.dt.day
    df['mmtt'] = df['month'].astype(str) + '_' + df['turbine_id']
    return df


if __name__ == '__main__':
    train = add_features(train)
    test = add_features(test)

    # save new features
    train.to_csv('../data/train_T1.csv', index=False)
    test.to_csv('../data/test_T1.csv', index=False)
    print('Done!')
