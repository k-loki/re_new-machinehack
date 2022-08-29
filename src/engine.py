# ONLY models go here...
import sys
sys.path.append(r'D:\Hacks\re_new')
import warnings
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor
from category_encoders.target_encoder import TargetEncoder
from sklearn.compose import make_column_transformer
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBRegressor

from src.utils import cat_cols, load_config

# load config
config = load_config()

def get_model(**kwargs):
    
    # model = CatBoostRegressor(random_seed=config['RAND'], n_estimators=5000, verbose=1000)
    # model = LGBMRegressor(random_state=config['RAND'], n_estimators=2000, n_jobs=-1, device='gpu', verbose=0, metric='mape')
    # model = XGBRegressor(tree_method='gpu_hist', random_state=config['RAND'], n_estimators=2000, n_jobs=-1, verbose=False)
    model = KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='auto', p=2, metric='minkowski', n_jobs=-1)

    tme = TargetEncoder()
    # pf = PolynomialFeatures(degree=2)
    # pca = PCA(n_components=10, random_state=config['RAND'])
    scaler = StandardScaler()
    ct = make_column_transformer(
        (tme, cat_cols),
        remainder = 'passthrough'
    )
    model_pipe = make_pipeline(
        ct,
        scaler,
        model
    )
    return model_pipe
