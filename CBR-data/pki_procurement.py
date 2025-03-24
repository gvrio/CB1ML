import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter

def computeFP(smiles, labels=None):
    moldata = [Chem.MolFromSmiles(mol) for mol in smiles]
    fpdata=[]
    for i, mol in enumerate(moldata):
        if mol:
            ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)]
            fpdata += [ecfp6]
    fp_df = pd.DataFrame(data=fpdata, index=smiles)
    if labels is not None: fp_df['labels'] =labels
    return fp_df

df = pd.read_csv('classification_data.csv')

    
df = df.dropna(subset=['param', 'value', 'unit'])
df = df[df['labels'] == 1]
pki_df = df[df['param']=='Ki']
pki_df = pki_df[pki_df['unit'] == 'nM']
pki_df['pki'] = [-np.log10(ki*10**-9) for ki in pki_df['value']]
pki_df.drop(['param', 'value', 'unit', 'labels'], axis=1, inplace=True)
pki_df.drop_duplicates(subset='smiles', keep='first', inplace=True)
pki_df.dropna(inplace=True)

pki_df = pki_df.sort_values(by='pki')
print(pki_df)
print(np.mean(pki_df['pki']))
print(np.median(pki_df['pki']))

pki_df.to_csv('regression_data.csv',index=False)
sns.histplot(data=pki_df['pki'], kde=True)
plt.show()

# fp_df = computeFP(list(df['smiles']), list(df['labels']))
# X = fp_df.drop('labels', axis=1)
# y = fp_df['labels']

# X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.7, stratify=y,random_state=999)

# model = SVC(random_state=999)
# model.fit(X_train, y_train)
# preds = model.predict(X_test)
# acc = accuracy_score(y_test, preds)
# f1 = f1_score(y_test, preds, average='binary')
# mcc = matthews_corrcoef(y_test, preds)
# print('acc', acc)
# print('f1', f1)
# print('mcc', mcc)