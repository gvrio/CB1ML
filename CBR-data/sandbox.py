import pandas as pd

class_df = pd.read_csv('regression_data.csv')

for source in ['CHEMBL218.tsv', 'P21554.tsv', '']:
    num = list(class_df['source']).count(source)
    print(source, num)