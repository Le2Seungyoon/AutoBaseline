import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder, RobustScaler, MinMaxScaler, StandardScaler

# encoding
def encode_categorical_features(train_x, test_x):
    """
    자동적으로 label encoding을 진행해주는 함수입니다. 

    Parameters:
    train_x (DataFrame): 학습 데이터의 feature 
    test_x (DataFrame): 테스트 데이터의 feature
    
    Returns:
    label encoding이 진행된 train_x와 train_y가 반환되게 됩니다. 
    """
    categorical_features = list(train_x.dtypes[train_x.dtypes == "object"].index)
    for i in categorical_features:
        le = LabelEncoder()
        le = le.fit(train_x[i])
        train_x[i] = le.transform(train_x[i])
        for case in np.unique(test_x[i]):
            if case not in le.classes_:
                le.classes_ = np.append(le.classes_, case)
        test_x[i] = le.transform(test_x[i])
    return train_x, test_x

# interpolation
def fill_missing_values(train_x, test_x, method='mean'):
    """
    학습 데이터와 테스트 데이터를 두 가지 방법으로 대치할 수 있는 함수입니다. 

    Parameters:
    train_x (DataFrame): 학습 데이터의 feature 
    test_x (DataFrame): 테스트 데이터의 feature
    method (str, optional): 대치 방법을 mean과 median 중에 정할 수 있습니다. 기본값은 mean입니다.

    Returns:
    결측값이 대치된 train_x와 test_x가 반환되게 됩니다. 
    """
    if method == 'median':
        train_x.fillna(train_x.median(), inplace=True)
        test_x.fillna(test_x.median(), inplace=True)
    elif method == 'mean':
        train_x.fillna(train_x.mean(), inplace=True)
        test_x.fillna(test_x.mean(), inplace=True)
    else:
        raise ValueError("Method must be 'median' or 'mean'.")
    return train_x, test_x

# scaling
def scale_features(train_x, test_x, scaling_target, method='robust'):
    """
    Scales specified features in train_x and test_x dataframes using the specified scaler type.

    Parameters:
    train_x (DataFrame): 학습 데이터의 feature 
    test_x (DataFrame): 테스트 데이터의 feature
    scaling_target (list): 스케일링을 적용할 컬럼을 선정합니다. 
    method (str, optional): 스케일링 방법을 standard, minmax, robust 중에 정할 수 있습니다. 기본값은 standard입니다.
    
    Returns:
    스케일링이 적용된 train_x와 test_x가 반환되게 됩니다. 
    """
    if method in ['robust', 'minmax', 'standard']:
        if method == 'robust':
            scaler = RobustScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        
        for i in scaling_target:
            train_data = train_x[i].values.reshape(-1, 1)
            test_data = test_x[i].values.reshape(-1, 1)
            scaler.fit(train_data)
            train_x[i] = scaler.transform(train_data)
            test_x[i] = scaler.transform(test_data)
    
    elif method == 'iqr':
        for i in scaling_target:
            Q1_train = train_x[i].quantile(0.25)
            Q3_train = train_x[i].quantile(0.75)
            
            IQR_train = Q3_train - Q1_train
            
            lower_bound_train = Q1_train - 1.5 * IQR_train
            upper_bound_train = Q3_train + 1.5 * IQR_train
            
            train_x[i] = np.where(train_x[i] < lower_bound_train, lower_bound_train, train_x[i])
            train_x[i] = np.where(train_x[i] > upper_bound_train, upper_bound_train, train_x[i])
            
            test_x[i] = np.where(test_x[i] < lower_bound_train, lower_bound_train, test_x[i])
            test_x[i] = np.where(test_x[i] > upper_bound_train, upper_bound_train, test_x[i])
    elif method == 'none':
        # Skip
        return train_x, test_x
    else:
        raise ValueError("Scaler type must be 'robust', 'minmax', 'standard', 'none', or 'iqr'.")
    
    return train_x, test_x

# to prevent error
def clean_column_names(dfs):
    """
    일부 특수문자에 의해 모델링 중 발생하는 에러를 방지하기 위한 코드입니다.
    """
    for df in dfs:
        if isinstance(df, pd.DataFrame):
            new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
            new_n_list = list(new_names.values())
            new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
            df.rename(columns=new_names, inplace=True)