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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from utils import MAE, RMSE, NMAE, R2, seed_everything

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
DATA_NAME = config['data']['name']
DATA_PATH = config['data']['data_path']
OUTPUT_PATH = config['data']['output_path']
MODELS = config['models']
METRICS = config['metrics']
SEED = config['seed']
TARGET = config['target']
FEATURE = config['feature']

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

# data setting
data = pd.read_csv(DATA_PATH + DATA_NAME)
data = data[FEATURE + [TARGET]]

## to prevent error1
categorical_features = list(data.dtypes[data.dtypes == "object"].index)
for i in categorical_features:
    le = LabelEncoder()
    le=le.fit(data[i])
    data[i]=le.transform(data[i])

train, test = train_test_split(data, train_size=0.7, random_state=SEED)
train_x = train.drop(columns=TARGET)
train_y = train[TARGET]
test_x = test.drop(columns=TARGET)
test_y = test[TARGET]

## to prevent error2
dfs = [train_x, train_y, test_x, test_y]
for df in dfs:
    if isinstance(df, pd.DataFrame):
        new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
        new_n_list = list(new_names.values())
        new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
        df.rename(columns=new_names, inplace=True)

if __name__ == '__main__':    
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