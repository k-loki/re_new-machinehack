import yaml
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error as mape

config_path = r'D:\Hacks\re_new\src\config.yaml'

def load_config(config_file=config_path):
    with open(config_file, 'r') as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config


config = load_config()


def load_data():
    train_df = pd.read_csv(config['train_data'])
    test_df = pd.read_csv(config['test_data'])
    return train_df, test_df


def comp_score(y_true, y_pred):
    return mape(y_true, y_pred)


cat_cols = ['turbine_id']
