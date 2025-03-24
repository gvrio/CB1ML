import time
import requests
import pandas as pd
import numpy as np
from rdkit import Chem
from collections import Counter

#load P21554.tsv retrieved from GLASS
GLASS_df = pd.read_csv('P21554.tsv', sep='\t')

#perform data cleaning steps
GLASS_df.columns = GLASS_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
GLASS_df.dropna(subset=['ligand_standard_inchi_key'],inplace=True)
GLASS_df.drop_duplicates(subset=['ligand_standard_inchi_key'],inplace=True)
GLASS_df['value'] = (GLASS_df['min'] + GLASS_df['max']) / 2
GLASS_df.drop(['glass#', 'uniprot', 'min', 'max','reference_pmid_or_other_citations'], axis=1, inplace=True)
print(GLASS_df)
print('\n--------------------------------------------------------------------------\n')

def inchi_to_smiles(inchi_lst, delay=0.3, verbose=True):
    #declare empty list to store SMILES from Pubchem
    smiles_list = []
    for i, inchi in enumerate(inchi_lst):
        
        #request pubchem servers and append smiles to list
        response = requests.get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{inchi}/property/CanonicalSMILES/TXT')
        smi = response.text
        smiles_list.append(smi)
        if verbose:
            print(i, smi)
        
        #script delay to prevent server overload
        time.sleep(delay)
    return smiles_list

pubchem_smiles = inchi_to_smiles()

#use rdkit to standardize smiles
standardized_smiles = []
for smi in pubchem_smiles:
    mol = Chem.MolFromSmiles(smi)
    std_smi = Chem.MolToSmiles(mol)
    standardized_smiles.append(std_smi)

#append smiles to df and drop any rows with invalid smiles
GLASS_df['smiles'] = standardized_smiles
GLASS_df.dropna(inplace=True)
GLASS_df.drop_duplicates(subset='smiles', inplace=True)

#append labels to df and save as csv
GLASS_df['labels'] = [1] * len(GLASS_df)
GLASS_df.to_csv('GLASS-data.csv', index=False)
print('\n--------------------------------------------------------------------------\n')
print(GLASS_df)


