import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from custom_funcs import computeFP

STATE = 111
files = ['new_svm.pkl', 'new_mlp.pkl', 'new_rf.pkl']

def load_data(file):
    df = pd.read_csv(file, index_col=False)
    fp_df = computeFP(list(df['smiles']), list(df['labels']))
    X = fp_df.drop('labels', axis=1)
    y = fp_df['labels']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, stratify=y, random_state=STATE)

    return (X_train,X_test, y_train, y_test)

def resample_and_scale(X_train, y_train):
    kmeans_smote = SMOTETomek(sampling_strategy='all', random_state=STATE)
    X_res, y_res = kmeans_smote.fit_resample(X_train, y_train)
    return (X_res.to_numpy(), y_res.to_numpy())

X_train, X_test, y_train, y_test = load_data('classification_data.csv')
X_res, y_res = resample_and_scale(X_train, y_train)
with open('results/ecfp6_class_results_0.pkl', 'rb') as f:
    results = pickle.load(f)
    for file, res in zip(files, results):
        model = res['model'](**res['cv results']['params'], random_state=STATE)
        model.fit(X_res, y_res)
        predictions = model.predict(X_test)
        print('acc: ',metrics.accuracy_score(y_test, predictions))
        print('f1: ',metrics.f1_score(y_test, predictions))
        print('mcc: ', metrics.matthews_corrcoef(y_test, predictions))
        c_matrix = metrics.confusion_matrix(y_test, predictions)
        tn, fp, fn, tp = c_matrix.ravel()
        print('tn: ', tn)
        print('fp: ', fp)
        print('fn: ', fn)
        print('tp: ', tp)
        with open(file, 'wb') as f:
            pickle.dump(model, f)
            print('Model saved')