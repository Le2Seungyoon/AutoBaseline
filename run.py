import pandas as pd
import numpy as np
import random
import os
import re
import yaml
import argparse
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm
import optuna
from utils import MAE, RMSE, NMAE, R2, seed_everything
from preprocessing import encode_categorical_features, fill_missing_values, scale_features, clean_column_names
from models.param_config import param_config

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='YAML 파일 지정', required=True)

args = parser.parse_args()
config_filename = args.config

# YAML 파일은 config 폴더 내부에 저장
config_directory = './config/'
config_path = config_directory + config_filename

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

SETTING = config_filename.replace('.yaml', '')
TRAIN_DATA_NAME = config['data']['train_data']
TEST_DATA_NAME = config['data']['test_data']
DATA_PATH = config['data']['data_path']
OUTPUT_PATH = config['data']['output_path']
MODELS = config['models']['active_model']
MODE = config['models']['mode']
MODEL_SAVE_PATH = config['models']['save_path']
TRIAL = config['models']['trial']
METRICS = config['metrics']
SEED = config['seed']
TARGET = config['target']
FEATURE = config['feature']
FILL_METHOD = config['fill_method']['method']
SCALE_TARGET = config['scale']['scale_target']
SCALE_METHOD = config['scale']['scale_method']

# seed setting
seed_everything(SEED)

# model define
lr = LinearRegression()
ridge = Ridge(random_state=SEED)
dt = DecisionTreeRegressor(random_state=SEED)
rf = RandomForestRegressor(random_state=SEED)
et = ExtraTreesRegressor(random_state=SEED)
xgb = XGBRegressor(tree_method='gpu_hist', gpu_id=0, random_state=SEED)
lgbm = LGBMRegressor(random_state=SEED)
cat = CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=SEED)

def main(): 
    # time setting
    t = pd.Timestamp.now()
    month = t.month
    day = t.day
    hour = t.hour
    minute = t.minute
    
    # data setting
    train = pd.read_csv(DATA_PATH + TRAIN_DATA_NAME)
    test = pd.read_csv(DATA_PATH + TEST_DATA_NAME)
    train_x = train[FEATURE]
    train_y = train[TARGET]
    test_x = test[FEATURE]
    test_y = test[TARGET]

    # preprocessing
    train_x, test_x = encode_categorical_features(train_x, test_x)
    train_x, test_x = fill_missing_values(train_x, test_x, method=FILL_METHOD)
    train_x, test_x = scale_features(train_x, test_x, scaling_target=SCALE_TARGET, method=SCALE_METHOD)
    dfs = [train_x, train_y, test_x, test_y]
    clean_column_names(dfs)
    
    # tuning setting
    def objective(trial):
        if model_name == 'lr':
            model = LinearRegression()

        elif model_name == 'ridge':
            alpha_min = param_config.ridge_config['parameters']['ALPHA']['min']
            alpha_max = param_config.ridge_config['parameters']['ALPHA']['max']
            params = {
                'alpha': trial.suggest_float('alpha', alpha_min, alpha_max),
                'random_state': SEED
            }
            model = Ridge(**params)

        elif model_name == 'dt':
            depth_min = param_config.dt_config['parameters']['MAX_DEPTH']['min']
            depth_max = param_config.dt_config['parameters']['MAX_DEPTH']['max']
            params = {
                'max_depth': trial.suggest_int('max_depth', depth_min, depth_max),
                'random_state': SEED
            }
            model = DecisionTreeRegressor(**params)

        elif model_name == 'rf':
            estimators_min = param_config.rf_config['parameters']['N_ESTIMATORS']['min']
            estimators_max = param_config.rf_config['parameters']['N_ESTIMATORS']['min']
            depth_min = param_config.rf_config['parameters']['MAX_DEPTH']['min']
            depth_max = param_config.rf_config['parameters']['MAX_DEPTH']['max']
            params = {
                'n_estimators': trial.suggest_int('n_estimators', estimators_min, estimators_max),
                'max_depth': trial.suggest_int('max_depth', depth_min, depth_max),
                'random_state': SEED
            }
            model = RandomForestRegressor(**params)

        elif model_name == 'et':
            estimators_min = param_config.et_config['parameters']['N_ESTIMATORS']['min']
            estimators_max = param_config.et_config['parameters']['N_ESTIMATORS']['min']
            depth_min = param_config.et_config['parameters']['MAX_DEPTH']['min']
            depth_max = param_config.et_config['parameters']['MAX_DEPTH']['max']
            params = {
                'n_estimators': trial.suggest_int('n_estimators', estimators_min, estimators_max),
                'max_depth': trial.suggest_int('max_depth', depth_min, depth_max),
                'random_state': SEED
            }
            model = ExtraTreesRegressor(**params)

        elif model_name == 'xgb':
            estimators_min = param_config.xgb_config['parameters']['N_ESTIMATORS']['min']
            estimators_max = param_config.xgb_config['parameters']['N_ESTIMATORS']['min']
            depth_min = param_config.xgb_config['parameters']['MAX_DEPTH']['min']
            depth_max = param_config.xgb_config['parameters']['MAX_DEPTH']['max']
            lr_min = param_config.xgb_config['parameters']['LEARNING_RATE']['min']
            lr_max = param_config.xgb_config['parameters']['LEARNING_RATE']['max']
            params = {
                'n_estimators': trial.suggest_int('n_estimators', estimators_min, estimators_max),
                'max_depth': trial.suggest_int('max_depth', depth_min, depth_max),
                'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max),
                'tree_method': 'gpu_hist', 
                'gpu_id': 0,
                'random_state': SEED
            }
            model = XGBRegressor(**params)

        elif model_name == 'lgbm':
            leaves_min = param_config.lgbm_config['parameters']['NUM_LEAVES']['min']
            leaves_max = param_config.lgbm_config['parameters']['NUM_LEAVES']['max']
            child_min = param_config.lgbm_config['parameters']['MIN_CHILD_SAMPLES']['min']
            child_max = param_config.lgbm_config['parameters']['MIN_CHILD_SAMPLES']['max']
            estimators_min = param_config.lgbm_config['parameters']['N_ESTIMATORS']['min']
            estimators_max = param_config.lgbm_config['parameters']['N_ESTIMATORS']['min']
            depth_min = param_config.lgbm_config['parameters']['MAX_DEPTH']['min']
            depth_max = param_config.lgbm_config['parameters']['MAX_DEPTH']['max']
            lr_min = param_config.lgbm_config['parameters']['LEARNING_RATE']['min']
            lr_max = param_config.lgbm_config['parameters']['LEARNING_RATE']['max']
            params = {
                'num_leaves': trial.suggest_int('num_leaves', leaves_min, leaves_max),
                'min_chiled_samples': trial.suggest_int('min_chiled_samples', child_min, child_max),
                'n_estimators': trial.suggest_int('n_estimators', estimators_min, estimators_max),
                'max_depth': trial.suggest_int('max_depth', depth_min, depth_max),
                'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max),
                'random_state': SEED
            }
            model = LGBMRegressor(**params)

        elif model_name == 'cat':
            iter_min = param_config.cat_config['parameters']['ITERATIONS']['min']
            iter_max = param_config.cat_config['parameters']['ITERATIONS']['min']
            depth_min = param_config.cat_config['parameters']['MAX_DEPTH']['min']
            depth_max = param_config.cat_config['parameters']['MAX_DEPTH']['max']
            lr_min = param_config.cat_config['parameters']['LEARNING_RATE']['min']
            lr_max = param_config.cat_config['parameters']['LEARNING_RATE']['max']
            params = {
                'iterations': trial.suggest_int('iterations', iter_min, iter_min),
                'max_depth': trial.suggest_int('max_depth', depth_min, depth_max),
                'learning_rate': trial.suggest_float('learning_rate', lr_min, lr_max),
                'verbose': False,
                'allow_writing_files': False,
                'random_state': SEED,
            }
            model = CatBoostRegressor(**params)

        model.fit(train_x, train_y)
        pred = model.predict(test_x)
        score = RMSE(test_y, pred)

        return score

    model_instances = [lr, ridge, dt, rf, et, xgb, lgbm, cat]
    metric_funcs = {'MAE': MAE, 'RMSE': RMSE, 'NMAE': NMAE, 'R2': R2}
    results = {'Metric': METRICS}
    
    for model_name, model in tqdm(zip(MODELS, model_instances), desc="Training Models"):
        if MODE == 'baseline':
            model.fit(train_x, train_y)
            pred = model.predict(test_x)
        elif MODE == 'tuning':
            print(model_name)
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=TRIAL)
            best_params = study.best_params
            best_model = model.set_params(**best_params)
            best_model.fit(train_x, train_y)
            pred = best_model.predict(test_x)
            # model save
            mname = f'{SETTING}_{model_name}_{month:02}{day:02}{hour:02}{minute:02}'
            pickle.dump(best_model, open(MODEL_SAVE_PATH + mname + '.model', 'wb'))
            with open(MODEL_SAVE_PATH + mname + '.yaml', 'w') as file:
                yaml.dump(best_model.get_params(), file)
        else:
            raise ValueError("Invalid mode. Please choose 'baseline' or 'tuning'.")
    
        model_metrics = []
        for metric_name in METRICS:
            metric_func = metric_funcs[metric_name]
            metric_value = metric_func(test_y, pred)
            model_metrics.append(metric_value)
    
        results[model_name] = model_metrics
    
    results_df = pd.DataFrame(results)
    
    # output save
    fname = f"{SETTING}_result_{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}.csv"
    results_df.to_csv(OUTPUT_PATH + fname, index=False)

if __name__ == '__main__':
    main()