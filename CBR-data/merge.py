import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from collections import Counter

def pairwise_similarity(smiles):
    
    fgrps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 3) for smi in smiles]
    print(list(fgrps[0]))
    print(len(list(fgrps[0])))
    nfgrps=len(fgrps)
    
    global similarities

    similarities = np.zeros((nfgrps, nfgrps))

    for i in range(1, nfgrps):
            similarity = DataStructs.BulkTanimotoSimilarity(fgrps[i], fgrps[:i])
            similarities[i, :i] = similarity
            similarities[:i, i] = similarity

    return similarities

def tanimoto_filtering(smiles_list, smiles_reference, threshold):
    # Calculate fingerprints for reference molecules
    ref_fps = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 3) for smi in smiles_reference]
    

    # Filter out similar molecules
    sim_lst = []
    filtered_smiles = []
    filtered_sim_lst = []
    for smi1 in smiles_list:
        mol = Chem.MolFromSmiles(smi1)
        fp1 = AllChem.GetMorganFingerprint(mol, 3)

        similarities = DataStructs.BulkTanimotoSimilarity(fp1, ref_fps)
        mean_similarity = np.mean(similarities)
        
        if mean_similarity <= threshold:
            filtered_smiles.append(smi1)
            filtered_sim_lst.append(mean_similarity)
        sim_lst.append(mean_similarity)
    return filtered_smiles, np.mean(filtered_sim_lst), np.mean(sim_lst)

###################################################################################################################################################

glass_df = pd.read_csv('GLASS-data.csv')
chembl_df = pd.read_csv('ChEMBL-data.csv')

chembl_actives = chembl_df[chembl_df['labels'] == 1]


active_smiles = []
active_smiles.extend(list(glass_df['smiles']))
active_smiles.extend(list(chembl_actives['smiles']))

active_params = []
active_params.extend(list(glass_df['param']))
active_params.extend(list(chembl_actives['param']))


active_values = []
active_values.extend(list(glass_df['value']))
active_values.extend(list(chembl_actives['value']))


active_units = []
active_units.extend(list(glass_df['unit']))
active_units.extend(list(chembl_actives['unit']))


active_labels = []
active_labels.extend(list(glass_df['labels']))
active_labels.extend(list(chembl_actives['labels']))


active_sources = []
active_sources.extend(len(glass_df) * ['GLASS'])
active_sources.extend(len(chembl_actives) * ['ChEMBL'])

active_data_dict = dict(
    zip(
        ['smiles', 'param', 'value', 'unit', 'labels', 'source'], 
        [active_smiles, active_params, active_values, active_units, active_labels, active_sources]
        )
    )

active_df = pd.DataFrame(data=active_data_dict)
active_df = active_df.sort_values(by=['source'])
active_df.drop_duplicates(keep='first', subset='smiles', inplace=True)
active_df.reset_index(inplace=True)
active_df.drop('index', axis=1, inplace=True)
source_count = dict(Counter(list(active_df['source'])))
label_count = dict(Counter(list(active_df['labels'])))
print('-----------------------------------------------------')
print('Active data')
print(active_df)
print(f'Source count: {source_count}')
print(f'Label count: {label_count}')
print('-----------------------------------------------------')

# mean_pairwise_similarity = np.mean(pairwise_similarity(list(active_df['smiles'])))
# threshold = np.round(mean_pairwise_similarity - 0.01, 2)
# print(f'Mean pairwise similarity for active ligands: {mean_pairwise_similarity}')
# print(f'Setting max similarity threshold to: {threshold}')


###################################################################################################################################################

chembl_inactives = chembl_df[chembl_df['labels'] == 0]
decoy_df = pd.read_csv('decoys.csv')


decoy_smiles = list(decoy_df['smiles'])
std_decoy_smiles = []
for smi in decoy_smiles:
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        continue
    else:
        std_smi = Chem.MolToSmiles(mol)
    std_decoy_smiles.append(std_smi)





# filtered_zinc_smiles, filtered_sim, total_sim = tanimoto_filtering(std_zinc_smiles, list(active_df['smiles']), threshold)
# print(f'Mean pairwise similarity between ZINC compounds and ligands: {total_sim}')
# print(f'Mean pairwise similarity between filtered compounds and ligands: {filtered_sim}')
# print('-----------------------------------------------------')


##adding inactive samples
inactive_smiles = []
inactive_smiles.extend(list(chembl_inactives['smiles']))
inactive_smiles.extend(std_decoy_smiles)


inactive_params = []
inactive_params.extend(list(chembl_inactives['param']))
inactive_params.extend(len(std_decoy_smiles) * ['NaN'])


inactive_values = []
inactive_values.extend(list(chembl_inactives['value']))
inactive_values.extend(len(std_decoy_smiles) * ['NaN'])


inactive_units = []
inactive_units.extend(list(chembl_inactives['unit']))
inactive_units.extend(len(std_decoy_smiles) * ['NaN'])


inactive_labels = []
inactive_labels.extend(list(chembl_inactives['labels']))
inactive_labels.extend(len(std_decoy_smiles) * [0])


inactive_sources = []
inactive_sources.extend(len(chembl_inactives) * ['ChEMBL'])
inactive_sources.extend(len(std_decoy_smiles) * ['GDD/Decoy'])


inactive_data_dict = dict(
    zip(
        ['smiles', 'param', 'value', 'unit', 'labels', 'source'], 
        [inactive_smiles, inactive_params, inactive_values, inactive_units, inactive_labels, inactive_sources]
        )
    )

inactive_df = pd.DataFrame(data=inactive_data_dict)
inactive_df = inactive_df.sort_values(by=['source'])
inactive_df.drop_duplicates(keep='first', subset='smiles', inplace=True)
inactive_df.reset_index(inplace=True)
inactive_df.drop('index', axis=1, inplace=True)

source_count = dict(Counter(list(inactive_df['source'])))
label_count = dict(Counter(list(inactive_df['labels'])))
print('-----------------------------------------------------')
print('Inactive data')
print(inactive_df)
print(f'Source count: {source_count}')
print(f'Label count: {label_count}')
print('-----------------------------------------------------')

merged_df = pd.concat([inactive_df, active_df], ignore_index=True)
merged_df = merged_df.sort_values(by=['labels', 'source'])
merged_df.drop_duplicates(keep='first', subset='smiles', inplace=True)
merged_df.reset_index(inplace=True)
merged_df.drop('index', axis=1, inplace=True)
merged_df.to_csv('classification_data.csv', index=False)


source_count = dict(Counter(list(merged_df['source'])))
label_count = dict(Counter(list(merged_df['labels'])))
print('-----------------------------------------------------')
print('Merged data')
print(merged_df)
print(f'Source count: {source_count}')
print(f'Label count: {label_count}')
print('-----------------------------------------------------')


# def pairwise_tanimoto(smiles):
#     moldata = [Chem.MolFromSmiles(smi) for smi in smiles]
#     rdkit_gen = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=7)
#     fpdata = [rdkit_gen.GetFingerprint(mol) for mol in moldata]
            
#     num_fps = len(fpdata)
#     similarities = np.zeros((num_fps, num_fps))
    
#     for i in range(1, num_fps):
#         similarity = DataStructs.BulkTanimotoSimilarity(fpdata[i], fpdata[:i])
#         similarities[i, :i] = similarity
#         similarities[:i, i] = similarity

#     return similarities

# similarity_matrix=pairwise_tanimoto(list(merged_df['smiles']))
# sns.heatmap(data=similarity_matrix)
# plt.show()