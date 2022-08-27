"""
generate folds for cross validation
"""

import sys
sys.path.append(r'D:\Hacks\re_new')
import warnings
warnings.filterwarnings('ignore')
from src.utils import load_config

import pandas as pd
from sklearn.model_selection import KFold


config = load_config()
randseed = config['RAND']

data = config['DATA_DIR']
train_path = data + '/train.csv'
train = pd.read_csv(train_path)

kf = KFold(n_splits=10, random_state=randseed, shuffle=True)
for fold, (_, val_idx) in enumerate(kf.split(train)):
    train.loc[val_idx, 'fold'] = fold

print(train.fold.value_counts(normalize=True))

train.to_csv('../data/train_folds.csv', index=False)

print('All Done bruv!!')