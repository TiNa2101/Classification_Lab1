import pandas as pd
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.multioutput import MultiOutputClassifier

def predict(model: MultiOutputClassifier,
            train_data: pd.DataFrame,
            val_data: pd.DataFrame,
            train_target: pd.DataFrame,
            val_target: pd.DataFrame):
    val_target_pred = model.predict(val_data)
    train_target_pred = model.predict(train_data)

    classif_train = multilabel_confusion_matrix(train_target_pred, train_target)
    TrueNegative = []
    FalsePositive = []
    FalseNegative = []
    TruePositive = []
    precision = []
    for matr in classif_train:
        tn, fp, fn, tp = matr.ravel()
        TrueNegative.append(tn)
        FalsePositive.append(fp)
        FalseNegative.append(fn)
        TruePositive.append(tp)
        precision.append(sum(TruePositive) / (sum(TruePositive) + sum(FalsePositive)))

    TrueNegative = []
    FalsePositive = []
    FalseNegative = []
    TruePositive = []
    recall = []
    classif_test = multilabel_confusion_matrix(val_target_pred, val_target)
    for matr in classif_test:
        tn, fp, fn, tp = matr.ravel()
        TrueNegative.append(tn)
        FalsePositive.append(fp)
        FalseNegative.append(fn)
        TruePositive.append(tp)
        recall.append(sum(TruePositive) / (sum(TruePositive) + sum(FalseNegative)))
    
    return precision, recall
