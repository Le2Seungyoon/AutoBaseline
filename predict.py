import pandas as pd
import numpy as np
import random
import os
import re
import yaml
import argparse
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from tqdm import tqdm
from utils import MAE, RMSE, NMAE, R2, seed_everything
from preprocessing import encode_categorical_features, fill_missing_values, scale_features, clean_column_names

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='YAML 파일 지정', required=True)

args = parser.parse_args()
config_filename = args.config

# YAML 파일은 config 폴더 내부에 저장
config_directory = './config/prediction/'
config_path = config_directory + config_filename

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

SETTING = config_filename.replace('.yaml', '')
TRAIN_DATA_NAME = config['data']['train_data']
TEST_DATA_NAME = config['data']['test_data']
DATA_PATH = config['data']['data_path']
OUTPUT_PATH = config['data']['output_path']
MODELS = config['models']
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

    model_instances = [lr, ridge, dt, rf, et, xgb, lgbm, cat]
    metric_funcs = {'MAE': MAE, 'RMSE': RMSE, 'NMAE': NMAE, 'R2': R2}

    results = {'Metric': METRICS}

    for model, name in tqdm(list(zip(model_instances, MODELS)), desc="Training Models"):
        model.fit(train_x, train_y)
        pred = model.predict(test_x)

        model_metrics = []
        for metric_name in METRICS:
            metric_func = metric_funcs[metric_name]
            metric_value = metric_func(test_y, pred)
            model_metrics.append(metric_value)

        results[name] = model_metrics

    results_df = pd.DataFrame(results)
    
    t = pd.Timestamp.now()
    fname = f"{SETTING}_result_{t.month:02}{t.day:02}{t.hour:02}{t.minute:02}.csv"
    results_df.to_csv(OUTPUT_PATH + fname, index=False)
    print('Done.')

if __name__ == '__main__':
    main()