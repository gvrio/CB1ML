import sys
import time
import pickle
import deepchem.feat 
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from functools import partial
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from datetime import datetime

rdkit_desc = [
            'MinAbsPartialCharge',
            'NumRadicalElectrons',
            'HeavyAtomMolWt',
            'MaxAbsEStateIndex',
            'MaxAbsPartialCharge',
            'MaxEStateIndex',
            'MinPartialCharge',
            'ExactMolWt',
            'MolWt',
            'NumValenceElectrons',
            'MinEStateIndex',
            'MinAbsEStateIndex',
            'MaxPartialCharge',
            'BalabanJ',
            'BertzCT',
            'Chi0',
            'Chi0n',
            'Chi0v',
            'Chi1',
            'Chi1n',
            'Chi1v',
            'Chi2n',
            'Chi2v',
            'Chi3n',
            'Chi3v',
            'Chi4n',
            'Chi4v',
            'HallKierAlpha',
            'Ipc',
            'Kappa1',
            'Kappa2',
            'Kappa3',
            'LabuteASA',
            'PEOE_VSA1',
            'PEOE_VSA10',
            'PEOE_VSA11',
            'PEOE_VSA12',
            'PEOE_VSA13',
            'PEOE_VSA14',
            'PEOE_VSA2',
            'PEOE_VSA3',
            'PEOE_VSA4',
            'PEOE_VSA5',
            'PEOE_VSA6',
            'PEOE_VSA7',
            'PEOE_VSA8',
            'PEOE_VSA9',
            'SMR_VSA1',
            'SMR_VSA10',
            'SMR_VSA2',
            'SMR_VSA3',
            'SMR_VSA4',
            'SMR_VSA5',
            'SMR_VSA6',
            'SMR_VSA7',
            'SMR_VSA8',
            'SMR_VSA9',
            'SlogP_VSA1',
            'SlogP_VSA10',
            'SlogP_VSA11',
            'SlogP_VSA12',
            'SlogP_VSA2',
            'SlogP_VSA3',
            'SlogP_VSA4',
            'SlogP_VSA5',
            'SlogP_VSA6',
            'SlogP_VSA7',
            'SlogP_VSA8',
            'SlogP_VSA9',
            'TPSA',
            'EState_VSA1',
            'EState_VSA10',
            'EState_VSA11',
            'EState_VSA2',
            'EState_VSA3',
            'EState_VSA4',
            'EState_VSA5',
            'EState_VSA6',
            'EState_VSA7',
            'EState_VSA8',
            'EState_VSA9',
            'VSA_EState1',
            'VSA_EState10',
            'VSA_EState2',
            'VSA_EState3',
            'VSA_EState4',
            'VSA_EState5',
            'VSA_EState6',
            'VSA_EState7',
            'VSA_EState8',
            'VSA_EState9',
            'FractionCSP3',
            'HeavyAtomCount',
            'NHOHCount',
            'NOCount',
            'NumAliphaticCarbocycles',
            'NumAliphaticHeterocycles',
            'NumAliphaticRings',
            'NumAromaticCarbocycles',
            'NumAromaticHeterocycles',
            'NumAromaticRings',
            'NumHAcceptors',
            'NumHDonors',
            'NumHeteroatoms',
            'NumRotatableBonds',
            'NumSaturatedCarbocycles',
            'NumSaturatedHeterocycles',
            'NumSaturatedRings',
            'RingCount',
            'MolLogP',
            'MolMR'
]

def timestamp():
    ts = time.time()
    date_time = datetime.fromtimestamp(ts)
    stamp_str = date_time.strftime("%d-%m-%Y, %H:%M:%S")
    return stamp_str


def computeFP(smiles, labels):
    moldata = [Chem.MolFromSmiles(mol) for mol in smiles]
    fpdata=[]
    for i, mol in enumerate(moldata):
        if mol is not None:
            ecfp6 = [int(x) for x in AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)]
            fpdata += [ecfp6]
        else:
            fpdata += [[None for i in range(2048)]]
    fp_df = pd.DataFrame(data=fpdata, index=smiles)
    fp_df['labels'] = labels
    return fp_df

def computeDesc(smiles, labels):
    featurizer = deepchem.feat.RDKitDescriptors()
    initiate = featurizer.featurize('C')
    columns = [name[0] for name in Descriptors.descList if name[0] in rdkit_desc]
    allowedInd = [i for i, desc in enumerate(Descriptors.descList) if desc[0] in rdkit_desc]
    desc_data = []
    for mol in smiles:
        features = featurizer.featurize(mol)
        allowedFeats = [feature for i, feature in enumerate(features[0]) if i in allowedInd]
        desc_data += [allowedFeats]
    descriptors_df = pd.DataFrame(data=desc_data,index=smiles,columns=columns)
    descriptors_df['labels'] = labels
    return descriptors_df

        
def kfold_validation(estimator,  output, x, y, ckp_file=None, partial=False):
    
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
    ckp = 0
    acc_lst, f1_lst, mcc_lst = [],[],[]
    for train_ind, test_ind in kfold.split(x,y):
        ckp += 1
        X_train, X_test = x[train_ind], x[test_ind]
        Y_train, Y_test = y[train_ind], y[test_ind]
        
        if partial==True:
            estimator.partial_fit(X_train,Y_train)
        else:
            estimator.fit(X_train,Y_train)
            
        predictions = estimator.predict(X_test)
        acc = accuracy_score(Y_test, predictions)
        f1 = f1_score(Y_test, predictions, average='macro')
        mcc = matthews_corrcoef(Y_test, predictions)
        
        acc_lst += [acc]
        f1_lst += [f1]
        mcc_lst += [mcc]

        time = timestamp()

        if ckp_file is not None:
            with open(ckp_file, 'a') as f_output:
                f_output.write(f"{time}\n{estimator}, ckp: {ckp}\n acc: {acc}\n f1: {f1}\n mcc: {mcc}\n\n" )

    avg_acc = np.mean(acc_lst)
    avg_f1 = np.mean(f1_lst)
    avg_mcc = np.mean(mcc_lst)
    time = timestamp()
   
    with open(output, 'a') as f_output:
            f_output.write(f"{time}\nFinal Metrics for {estimator} \n Average acc: {avg_acc}\n Average F1: {avg_f1}\n Average MCC: {avg_mcc}\n\n")

    return estimator


def train_split_validation(estimator, X, y):
    
    acc_lst, f1_lst, mcc_lst = [],[],[]
    
    random_states = [0, 333, 555, 777, 999]
    
    for state in random_states:
        
        X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                            train_size=0.82,
                                                            test_size=0.18, 
                                                            random_state=state, 
                                                            stratify=y
                                                            )
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        f1 = f1_score(y_test, predictions, average='macro')
        mcc = matthews_corrcoef(y_test, predictions)
        
        acc_lst += [acc]
        f1_lst += [f1]
        mcc_lst += [mcc]

        
        
    avg_acc = np.mean(acc_lst)
    avg_f1 = np.mean(f1_lst)
    avg_mcc = np.mean(mcc_lst)
    time = timestamp()
    
    results = {'acc': avg_acc, 'f1': avg_f1, 'mcc': avg_mcc }
    
    return f'time: {time} \n results: {results} \n'
    

def optimize_params(estimators, spaces, tuner, x, y, cv=5, split=1500, output=None):
    
    opt_hyperparams = []
    
    X_opt, Y_opt = x[:split], y[:split]
    
    
    search_grids = spaces['grids']
    search_spaces = spaces['bayes']
    
    keys = list(estimators.keys())
    
    if tuner == 'grid':
        for key, ind in zip(keys, range(len(keys))):
            model = estimators[key]
            model_params = search_grids[ind]
            
            search = GridSearchCV(
                estimator=model(),
                param_grid=model_params,
                cv=cv,
                verbose=False
            )
            
            search.fit(X_opt,Y_opt)
            opt_hp = search.best_params_
            opt_hyperparams += [opt_hp]
            
            time = timestamp()
            print(time)
            if output is not None:
                with open(output, 'a') as f_output:
                    f_output.write(f'{time}\n{model} {tuner} optimized hyperparamters: \n {opt_hp}\n\n')
                    
        return {tuner: dict(zip(keys, opt_hyperparams))}
    
    else:
        print('tuner must be grid or bayes')
        return None
    
def pairwise_tanimoto_filtering(smiles=list, reference_smiles=list, threshold=float):
    result = smiles.copy()
    for smi1 in smiles:
        tanimoto_lst = []
        mol = Chem.MolFromSmiles(smi1)
        fp1 = AllChem.GetMorganFingerprint(mol, 3)
        for smi2 in reference_smiles:
            ref_mol = Chem.MolFromSmiles(smi2)
            fp2 = AllChem.GetMorganFingerprint(ref_mol, 3)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            tanimoto_lst.append(similarity)
        mean_similarity = np.mean(tanimoto_lst)
        if mean_similarity > threshold:
            result.remove(smi1)
            print(f'Removed {smi1}\nMean similarity: {mean_similarity}')
    return result

def smiles_standardization(smiles):
    std_smiles = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            std_smi = None
        else:
            std_smi = Chem.MolToSmiles(mol)
        std_smiles.append(std_smi)
    return std_smiles

def standardize_columns(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    return df