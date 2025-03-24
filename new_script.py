import pickle
import pandas as pd
import numpy as np
import statistics as sts
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from functools import partial
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll.base import scope
from imblearn.metrics import geometric_mean_score
from imblearn.combine import SMOTETomek
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

df = pd.read_csv('data/classification_data.csv')

def calculate_tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    fp1 = AllChem.GetMorganFingerprint(mol1, 2)
    fp2 = AllChem.GetMorganFingerprint(mol2, 2)
    return DataStructs.TanimotoSimilarity(fp1, fp2)

