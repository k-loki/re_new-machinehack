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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

# load config
config = load_config()

def get_model(**kwargs):

    model = LGBMRegressor(random_state=config['RAND'], n_estimators=2000, n_jobs=-1, device='gpu', verbose=0, metric='mape')
    tme = TargetEncoder()
    # pf = PolynomialFeatures(degree=2)
    # pca = PCA(n_components=10, random_state=config['RAND'])
    # scaler = StandardScaler()
    ct = make_column_transformer(
        (tme, cat_cols),
        remainder = 'passthrough'
    )
    model_pipe = make_pipeline(
        ct,
        model
    )
    return model_pipe
