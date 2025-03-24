import pandas as pd
import numpy as np
from collections import Counter

glass_df = pd.read_csv('P21554.tsv', sep='\t')
chembl_df = pd.read_csv('CHEMBL218.tsv', sep='\t')
decoy_df = pd.read_csv('decoys.csv')
zinc_df = pd.read_csv('in-trials.csv')


class_df = pd.read_csv('classification_data.csv')
reg_df = pd.read_csv('regression_data.csv')


# for name, df in zip(['GLASS', 'CHEMBL', 'GDD', 'ZINC'], [glass_df,chembl_df,decoy_df,zinc_df]):
#     print(name, len(df))

print('classification data count', len(class_df))
print(Counter(class_df['labels']))
print('regression data count', len(reg_df))
print(np.mean(reg_df['pki']))
    