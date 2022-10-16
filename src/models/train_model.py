import pandas as pd
from catboost import CatBoostClassifier
from sklearn.multioutput import MultiOutputClassifier


def train(train_data: pd.DataFrame, train_target: pd.DataFrame) -> MultiOutputClassifier:
    cat_model = CatBoostClassifier(n_estimators=100,
                                   depth=8,
                                   learning_rate=1, 
                                   loss_function='Logloss')
    model = MultiOutputClassifier(cat_model, n_jobs=1)
    model.fit(train_data, train_target)
    
    return model
