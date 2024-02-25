ridge_config = {
    'name': 'ridge',
    'method': 'bayes',
    'metric': {
        'name': 'rmse',
        'goal': 'minimize'   
    },
    'parameters': {
        'ALPHA': {
            'min': 0,
            'max': 10
        }
    }
}

dt_config = {
    'name': 'decisiontree',
    'method': 'bayes',
    'metric': {
        'name': 'rmse',
        'goal': 'minimize'   
    },
    'parameters': {
        'MAX_DEPTH': {
            'min': 5,
            'max': 100
        }
    }
}


rf_config = {
    'name': 'randomforest',
    'method': 'bayes',
    'metric': {
        'name': 'rmse',
        'goal': 'minimize'   
    },
    'parameters': {
        'N_ESTIMATORS': {
            'min': 50,
            'max': 1000
        },
        'MAX_DEPTH': {
            'min': 5,
            'max': 100
        }
    }
}

et_config = {
    'name': 'extratrees',
    'method': 'bayes',
    'metric': {
        'name': 'rmse',
        'goal': 'minimize'   
    },
    'parameters': {
        'N_ESTIMATORS': {
            'min': 50,
            'max': 1000
        },
        'MAX_DEPTH': {
            'min': 5,
            'max': 100
        }
    }
}

xgb_config = {
    'name': 'xgboost',
    'method': 'bayes',
    'metric': {
        'name': 'rmse',
        'goal': 'minimize'   
    },
    'parameters': {
        'N_ESTIMATORS': {
            'min': 50,
            'max': 1000
        },
        'MAX_DEPTH': {
            'min': 3,
            'max': 12
        },
        'LEARNING_RATE': {
            'min': 0.001,
            'max': 0.1
        }
    }
}

lgbm_config = {
    'name': 'lgbm',
    'method': 'bayes',
    'metric': {
        'name': 'rmse',
        'goal': 'minimize'   
    },
    'parameters': {
        'NUM_LEAVES': {
            'min': 8,
            'max': 256
        },
        'MIN_CHILD_SAMPLES': {
            'min': 20,
            'max': 1000
        },
        'N_ESTIMATORS': {
            'min': 50,
            'max': 1000
        },
        'MAX_DEPTH': {
            'min': 3,
            'max': 12
        },
        'LEARNING_RATE': {
            'min': 0.001,
            'max': 0.1
        }
    }
}

cat_config = {
    'name': 'catboost',
    'method': 'bayes',
    'metric': {
        'name': 'rmse',
        'goal': 'minimize'   
    },
    'parameters': {
        'ITERATIONS': {
            'min': 50,
            'max': 1000
        },
        'MAX_DEPTH': {
            'min': 4,
            'max': 10
        },
        'LEARNING_RATE': {
            'min': 0.001,
            'max': 0.1
        }
    }
}