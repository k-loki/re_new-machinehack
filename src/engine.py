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
from sklearn.neighbors import KNeighborsRegressor

# load config
config = load_config()

def get_model(**kwargs):

    model = KNeighborsRegressor(n_neighbors=55, weights='distance', algorithm='auto', p=2, metric='minkowski', n_jobs=-1)
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
