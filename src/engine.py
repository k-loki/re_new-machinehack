# ONLY models go here...
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
from sklearn.ensemble import RandomForestRegressor

# load config
config = load_config()


def get_model(**kwargs):

    model = CatBoostRegressor(n_estimators=5000, verbose=1000, task_type='GPU', random_seed=config['RAND'])
    # model = RandomForestRegressor(n_estimators=200, verbose=True, random_state=config['RAND'])
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
