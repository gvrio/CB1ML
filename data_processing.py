import pandas as pd
import matplotlib.pyplot as plt
import warnings



# with open('cb1-ligands.txt') as file:
#     lines = file.readlines()
#     print(lines)
ligand_df = pd.read_csv('target_ligand_data (1).csv')
decoy_df = pd.read_csv('cb1-decoys.csv')
ligand_df.columns = ligand_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
decoy_df.rename(columns={'zinc_id': 'name'}, inplace=True)

decoy_names = list(decoy_df['name'])
decoy_smiles = list(decoy_df['smiles'])
ligand_smiles = list(ligand_df['smiles'])
decoy_smiles_filter = [smiles for smiles in decoy_smiles if smiles not in ligand_smiles]


sorted_ligand_df = ligand_df.sort_values(by='name')
sorted_ligand_df.drop(['unnamed:_16', 'unnamed:_17', 'unnamed:_18',
       'unnamed:_19', 'unnamed:_20', 'unnamed:_21', 'unnamed:_22',
       'unnamed:_23', 'rot_bonds', 'h_dons', 'h_acc', 'name'], axis=1, inplace=True)

pki_df = sorted_ligand_df[sorted_ligand_df.unit == 'pKi']
pki_df.drop_duplicates(subset=['smiles'],keep='first',inplace=True)
pki_df.dropna(subset=['smiles', 'value'], inplace=True)
pki_df.reset_index(inplace=True, drop=True)
print(pki_df)

pki_df.to_csv('pki-ligands.csv', index=False)
