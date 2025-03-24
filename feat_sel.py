import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef
from custom_funcs import computeDesc, computeFP

df = pd.read_csv('CBR-data/classification_data.csv')
df = computeFP(list(df['smiles']), list(df['labels']))
df.to_csv('ecfp6_data_classification.csv', index=False)
X = np.array(df.drop('labels', axis=1))
y = np.array(df['labels'])

features = list(df.drop('labels', axis=1).columns)


kfold = StratifiedKFold(5)

accuracies = []
importances = []
for idx in kfold.split(X,y):
    train_idx, test_idx = idx[0], idx[1]
    xtrain, xtest = X[train_idx], X[test_idx]
    ytrain, ytest = y[train_idx], y[test_idx]
    rf_feat_sel = RandomForestClassifier(random_state=999)
    rf_feat_sel.fit(xtrain,ytrain)
    preds = rf_feat_sel.predict(xtest)
    mcc = matthews_corrcoef(ytest,preds)
    fold_mdis = rf_feat_sel.feature_importances_
    accuracies.append(mcc)
    importances.append(fold_mdis)
importances_df = pd.DataFrame(importances, columns=features)
importances_df['mcc'] = accuracies

print(importances_df)