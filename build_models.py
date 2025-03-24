##BUILD SKLEARN MODELS FOR STREAMLIT APPLICATION
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from custom_funcs import computeFP
from rdkit import Chem
from rdkit.Chem import AllChem

# STATE= 999

# class_ecfp6_results5 = [
#     {'model': 'sklearn.svm._classes.SVC', 'validation results': {'acc': 0.9842767295597484, 'f1': 0.9911426040744021, 'mcc': 0.9230777144462924, 'gmean': 0.9376690553602467, 'tn': 133, 'fp': 18, 'fn': 2, 'tp': 1119}, 'cv results': {'acc': 0.9954066985645933, 'f1': 0.9953797891637821, 'mcc': 0.9908632527096474, 'gmean': 0.995394706727692, 'params': {'C': 272.42516963297186}}}, 
#     {'model': 'sklearn.neural_network._multilayer_perceptron.MLPClassifier', 'validation results': {'acc': 0.9795597484276729, 'f1': 0.9884444444444445, 'mcc': 0.9004449306706707, 'gmean': 0.9382390679539139, 'tn': 134, 'fp': 17, 'fn': 9, 'tp': 1112}, 'cv results': {'acc': 0.993301984319394, 'f1': 0.993258241321036, 'mcc': 0.9866769806628474, 'gmean': 0.9932839552063187, 'params': {'activation': 'relu', 'hidden_layer_sizes': 1300}}}, 
#     {'model': 'sklearn.ensemble._forest.RandomForestClassifier', 'validation results': {'acc': 0.9724842767295597, 'f1': 0.984561093956771, 'mcc': 0.8627759184168915, 'gmean': 0.8931687147366212, 'tn': 121, 'fp': 30, 'fn': 5, 'tp': 1116}, 'cv results': {'acc': 0.9823945401483895, 'f1': 0.9825575517592122, 'mcc': 0.9649825660404912, 'gmean': 0.9823449037720848, 'params': {'max_depth': 14, 'n_estimators': 898}}}
#     ]

# print(class_ecfp6_results5[0]['cv results']['params'])
# print(class_ecfp6_results5[1]['cv results']['params'])
# print(class_ecfp6_results5[2]['cv results']['params'])

# decoy_df = pd.read_csv('data/cb1-decoys.csv')
# test_ligands_df = pd.read_csv('data/test-ligands.csv')
# ligand_df = pd.read_csv('data/processed-ligands.csv')

# test_smiles = list(test_ligands_df['smiles'])
# test_labels = list(test_ligands_df['label'])
# ligand_df = ligand_df[~ligand_df['smiles'].isin(test_smiles)]
# decoy_df = decoy_df[~decoy_df['smiles'].isin(test_smiles)]

# train_labels = [1 for i in range(len(ligand_df))]
# train_labels.extend([0 for i in range(len(decoy_df))])
# train_smiles = list(ligand_df['smiles'])
# train_smiles.extend(list(decoy_df['smiles']))
# train_ecfp6_df = computeFP(train_smiles, train_labels)
# train_ecfp6_df.dropna(inplace=True)
# print('training size',len(train_ecfp6_df))

# X_train = train_ecfp6_df.drop('labels', axis=1)
# y_train = train_ecfp6_df['labels']

# # test_ecfp6_df = computeFP(test_smiles, test_labels)
# # X_test = test_ecfp6_df.drop('labels', axis=1)
# # y_test = test_ecfp6_df['labels']

# smote = SMOTETomek(sampling_strategy='all', random_state=STATE)
# X_res, y_res = smote.fit_resample(X_train, y_train)
# print('res training size',len(y_res))

# svm_model = SVC(**class_ecfp6_results5[0]['cv results']['params'], random_state=STATE)
# mlp_model = MLPClassifier(**class_ecfp6_results5[1]['cv results']['params'], random_state=STATE)
# rf_model = RandomForestClassifier(**class_ecfp6_results5[2]['cv results']['params'], random_state=STATE)

# # models = [svm_model, mlp_model, rf_model]
# # names = ['svm.pkl', 'mlp.pkl', 'rf.pkl']
# # for name, model in zip(names, models):
# #     model.fit(X_res,y_res)
# #     preds = model.predict(X_test)
# #     accs = metrics.accuracy_score(preds, y_test)
# #     print(name, accs)
# #     print(list(y_test))
# #     print(list(preds))
# #     with open(f'built_models/{name}', 'wb') as f:
# #         pickle.dump(model, f)
# #         print(f'{name} dumped.')


# for estimator in ['mlp.pkl', 'rf.pkl', 'svm.pkl']:
#     with open(f'built_models/{estimator}', 'rb') as f:
#         mol = Chem.MolFromSmiles('Cc3c(C(=O)NN1CCCCC1)nn(c2ccc(Cl)cc2Cl)c3c4ccc(I)cc4')
#         ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)]
#         model = pickle.load(f)
#         res = model.predict([ecfp6])
#         print(f'{estimator}: ',res)





