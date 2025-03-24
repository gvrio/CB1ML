import pandas as pd
import numpy as np
from rdkit import Chem
from collections import Counter

chembl_df = pd.read_csv('CHEMBL218.tsv', sep='\t')

chembl_df.columns = chembl_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
chembl_df.drop(['molecule_name', 'molecule_max_phase', 'molecular_weight', '#ro5_violations', 'alogp', 'compound_key', 'pchembl_value', 'data_validity_comment', 'uo_units', 'ligand_efficiency_bei', 'ligand_efficiency_le', 'ligand_efficiency_lle', 'ligand_efficiency_sei', 'potential_duplicate', 'assay_chembl_id', 'assay_description', 'assay_type', 'bao_format_id', 'bao_label', 'assay_organism', 'assay_tissue_chembl_id', 'assay_tissue_name', 'assay_cell_type', 'assay_subcellular_fraction', 'assay_parameters', 'assay_variant_accession', 'assay_variant_mutation', 'target_chembl_id', 'target_name', 'target_organism', 'target_type', 'document_chembl_id', 'source_id', 'source_description', 'document_journal', 'document_year', 'cell_chembl_id', 'properties'], axis=1, inplace=True)
chembl_df = chembl_df.sort_values(by=['standard_type'])
std_types = Counter(chembl_df['standard_type'])
sorted_std_types = sorted(std_types, key=lambda x: std_types[x], reverse=True)
chembl_df['standard_type_index'] = chembl_df['standard_type'].apply(lambda x: sorted_std_types.index(x))
chembl_df = chembl_df.sort_values(by=['standard_type_index', 'standard_value'])

standardized_smiles = []
for smi in list(chembl_df['smiles']):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        std_smi = None
    else:
        std_smi = Chem.MolToSmiles(mol)
    standardized_smiles.append(std_smi)

chembl_df['smiles'] = standardized_smiles
chembl_df.dropna(subset='smiles',inplace=True)
chembl_df.drop_duplicates(subset='smiles', keep='first',inplace=True)


labels = []
for row in chembl_df.itertuples():
    comment = str(row.comment).lower()
    if row.standard_type == 'IC50' or row.standard_type == 'EC50':
        if row.standard_value >= 32000:
            label = 0
        else:
            label = 1
    if 'not active' in comment or 'inactive' in comment:
        label = 0

    else:
        label = 1
    labels.append(label)



print(chembl_df)
print(Counter(chembl_df['standard_type']))
print(Counter(labels))

chembl_df['labels'] = labels
chembl_df.drop(['comment', 'standard_type_index'], axis=1, inplace=True)

chembl_df.to_csv('ChEMBL-data.csv', index=False)

