"""
Here all the preprocessed and feature-engineered data is collected 
and models collected and ran accoriding to the strategy
all the runs are tracked by wandb
all the run models are saved in ../models/
Also save cross validated preds (for ensembling down the path)

"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import argparse
import wandb
from joblib import dump
import pandas as pd
from utils import load_config, comp_score
from engine import get_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import time


wandb.login()

# Load configuration
config = load_config()
model_config = config["MODEL"]
print(model_config['model_name'])
print(model_config['desc'])
print(get_model())

# Load data
train_data_path = config['train_data']
train_data = pd.read_csv(train_data_path)
test_data = pd.read_csv(config['test_data'])


def train_and_eval(X_train, y_train, X_val, y_val):
    print('Training Model...')
    model = get_model()
    model.fit(X_train, y_train)
    train_score = comp_score(model.predict(X_train), y_train)
    print("Training MAPE: ", train_score)

    print('Validating Model..')
    preds = model.predict(X_val)
    val_score = comp_score(y_val, preds)
    print("Validation MAPE: ", val_score)
    print("validation rmse: ", mean_squared_error(y_val, preds, squared=False))

    return model, train_score, val_score


def __cross_validate(holdout=False, cv_predict=False, wandb_track=True):
    cv_scores = []

    drop_cols = ['timestamp', 'Target']
    
    if cv_predict:
        cvpreds_test = np.zeros(shape=(len(test_data), config['N_FOLDS']))
        cvpreds_train = np.zeros(shape=(len(train_data)))
    
    kf = KFold(n_splits=config['N_FOLDS'], random_state=config['RAND'], shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f'Fold : {fold}')

        train_fold = train_data.iloc[train_idx]
        val_fold = train_data.iloc[val_idx]

        X_train, y_train = train_fold.drop(
            drop_cols, axis=1), train_fold.Target
        X_val, y_val = val_fold.drop(
            drop_cols, axis=1), val_fold.Target
        
        start = time.time() 
        model, train_score, val_score = train_and_eval(
            X_train, y_train, X_val, y_val)
        end = time.time()
        print(f'Time taken: {end - start}')

        if wandb_track:
            # wandb tracking
            wandb.log({
                'fold': fold,
                'Train_score': train_score,
                'Val_score': val_score 
            })

        cv_scores.append(val_score)

        if cv_predict:
            # save predictions for ensembling
            cvpreds_test[:, fold] = model.predict(test_data.drop(['timestamp'], axis=1))
            cvpreds_train[val_idx] = model.predict(X_val)
            
        print('----------------------------------------------------------')

        # save_model
        if model_config["save_models"] :
            dump(model, model_config['save_model_to'] + '/' +  model_config['model_name'] + '_' + str(fold))
            print('Model saved')

        if holdout == True:
            break

    if cv_predict:
        print('Saving cross validated predictions...')
        test_cv = pd.DataFrame(cvpreds_test.mean(axis=1), columns=['Target'])
        train_cv = pd.DataFrame(cvpreds_train, columns=['Target'])
        print('Test shape: ', test_cv.shape)
        print('Train shape: ', train_cv.shape)
        test_cv.to_csv(f"../submissions/{model_config['model_name']}_test_cv.csv", index=False)
        train_cv.to_csv(f"../submissions/{model_config['model_name']}_train_cv.csv", index=False)

    print("AVG mape :", np.array(cv_scores).mean())


def cross_validate(holdout=False, wandb_track=True, cv_predict=False):
    if wandb_track:
        # wandb tracking
        with wandb.init(project=config['project'], name=model_config['model_name'], config=config):
            __cross_validate(holdout, wandb_track=wandb_track, cv_predict=cv_predict)
    else:
        __cross_validate(holdout, wandb_track=wandb_track, cv_predict=cv_predict)
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--holdout', action='store_true')
    parser.add_argument('--wandb_track', type=bool, default=False)
    parser.add_argument('--cv_predict', type=bool, default=False,
                        help='do u want to save each folds predictions')
    args = parser.parse_args()

    cross_validate(args.holdout, args.wandb_track, cv_predict=args.cv_predict)
