# ONLY models go here...
from math import remainder
import sys
sys.path.append(r'D:\Hacks\re_new')

import warnings
warnings.filterwarnings("ignore")

from src.utils import load_config
from src.utils import cat_cols
from category_encoders.target_encoder import TargetEncoder
from catboost import CatBoostRegressor
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

# load config
config = load_config()


def get_model(**kwargs):

    model = CatBoostRegressor(n_estimators=2000, verbose=1000, task_type='GPU')
    tme = TargetEncoder()
    ct = make_column_transformer(
        (tme, cat_cols),
        remainder = 'passthrough'
    )
    model_pipe = make_pipeline(
        tme,
        model
    )
    return model_pipe
