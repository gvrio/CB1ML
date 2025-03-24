import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.manifold import TSNE 
from custom_funcs import computeDesc
import seaborn as sns


class_rdkit_results1 = pickle.load(open('CBR-data/results/rdkit_class_results_0.pkl', 'rb'))
class_rdkit_results2 = pickle.load(open('CBR-data/results/rdkit_class_results_0.25.pkl', 'rb'))
class_rdkit_results3 = pickle.load(open('CBR-data/results/rdkit_class_results_0.5.pkl', 'rb'))
class_rdkit_results4 = pickle.load(open('CBR-data/results/rdkit_class_results_0.75.pkl', 'rb'))
class_rdkit_results5 = pickle.load(open('CBR-data/results/rdkit_class_results_1.pkl', 'rb'))

rdkit_results = [class_rdkit_results1, class_rdkit_results2, class_rdkit_results3, class_rdkit_results4, class_rdkit_results5]

class_ecfp6_results1 = pickle.load(open('CBR-data/results/ecfp6_class_results_0.pkl', 'rb'))
class_ecfp6_results2 = pickle.load(open('CBR-data/results/ecfp6_class_results_0.25.pkl', 'rb'))
class_ecfp6_results3 = pickle.load(open('CBR-data/results/ecfp6_class_results_0.5.pkl', 'rb'))
class_ecfp6_results4 = pickle.load(open('CBR-data/results/ecfp6_class_results_0.75.pkl', 'rb'))
class_ecfp6_results5 = pickle.load(open('CBR-data/results/ecfp6_class_results_1.pkl', 'rb'))

ecfp6_results = [class_ecfp6_results1, class_ecfp6_results2, class_ecfp6_results3, class_ecfp6_results4, class_ecfp6_results5]

## Transforming data for plotting
svm_rdkit_acc = [rdkit_results[i][0]['validation results']['acc'] for i in range(5)]
mlp_rdkit_acc = [rdkit_results[i][1]['validation results']['acc'] for i in range(5)]
rf_rdkit_acc = [rdkit_results[i][2]['validation results']['acc'] for i in range(5)]

svm_ecfp6_acc = [ecfp6_results[i][0]['validation results']['acc'] for i in range(5)]
mlp_ecfp6_acc = [ecfp6_results[i][1]['validation results']['acc'] for i in range(5)]
rf_ecfp6_acc = [ecfp6_results[i][2]['validation results']['acc'] for i in range(5)]

svm_rdkit_f1 = [rdkit_results[i][0]['validation results']['gmean'] for i in range(5)]
mlp_rdkit_f1 = [rdkit_results[i][1]['validation results']['gmean'] for i in range(5)]
rf_rdkit_f1 = [rdkit_results[i][2]['validation results']['gmean'] for i in range(5)]

svm_ecfp6_f1 = [ecfp6_results[i][0]['validation results']['gmean'] for i in range(5)]
mlp_ecfp6_f1 = [ecfp6_results[i][1]['validation results']['gmean'] for i in range(5)]
rf_ecfp6_f1 = [ecfp6_results[i][2]['validation results']['gmean'] for i in range(5)]

svm_rdkit_mcc = [rdkit_results[i][0]['validation results']['mcc'] for i in range(5)]
mlp_rdkit_mcc = [rdkit_results[i][1]['validation results']['mcc'] for i in range(5)]
rf_rdkit_mcc = [rdkit_results[i][2]['validation results']['mcc'] for i in range(5)]

svm_ecfp6_mcc = [ecfp6_results[i][0]['validation results']['mcc'] for i in range(5)]
mlp_ecfp6_mcc = [ecfp6_results[i][1]['validation results']['mcc'] for i in range(5)]
rf_ecfp6_mcc = [ecfp6_results[i][2]['validation results']['mcc'] for i in range(5)]

##build heat map arrays
acc_arr = [svm_rdkit_acc, svm_ecfp6_acc,  mlp_rdkit_acc, mlp_ecfp6_acc, rf_rdkit_acc, rf_ecfp6_acc]
f1_arr = [svm_rdkit_f1, svm_ecfp6_f1,  mlp_rdkit_f1, mlp_ecfp6_f1, rf_rdkit_f1, rf_ecfp6_f1]
mcc_arr = [svm_rdkit_mcc, svm_ecfp6_mcc,  mlp_rdkit_mcc, mlp_ecfp6_mcc, rf_rdkit_mcc, rf_ecfp6_mcc]

xticklabels = [50, 62.5, 75, 87.5, 100]
yticklabels = ['SVM RDKit', 'SVM ECFP6', 'MLP RDKit', 'MLP ECFP6', 'RF RDKit', 'RF ECFP6']
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = plt.get_cmap('twilight_shifted')
new_cmap = truncate_colormap(cmap, 0.65, 0.96)
#acc_heatmap = sns.heatmap(acc_arr, xticklabels=xticklabels, yticklabels=yticklabels,cmap=new_cmap,vmin=0.25,vmax=0.98,annot=True).set(title='Model evaluation results: Accuracy', xlabel='Features (%)', ylabel='Model')
#mcc_heatmap = sns.heatmap(mcc_arr, xticklabels=xticklabels, yticklabels=yticklabels,cmap=new_cmap,vmin=0.25, vmax=0.98,annot=True).set(title='Model evaluation results: Matthews Correlation Coef.', xlabel='Features (%)', ylabel='Model')
# f1_heatmap = sns.heatmap(f1_arr, xticklabels=xticklabels, yticklabels=yticklabels,cmap=new_cmap,vmin=0.25,vmax=0.98,annot=True).set(title='Model evaluation results: G-Mean', xlabel='Features (%)', ylabel='Model')
# plt.show()


def tsne_plot(data):
    tsne = TSNE(n_components=3, random_state=42)
    tsne_data = tsne.fit_transform(data)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tsne_data[:, 0], tsne_data[:, 1], tsne_data[:, 2])
    ax.set_xlabel('tSNE-x')
    ax.set_ylabel('tSNE-y')
    ax.set_zlabel('tSNE-z')
    ax.set_title('t-SNE Plot for Classification Data')
    plt.show()

df = pd.read_csv('CBR-data/classification_data.csv')
arr = np.array(df.drop('labels', axis=1))
rdkit_df = computeDesc(list(df['smiles']), list(df['labels']))
print(rdkit_df)
tsne_plot(rdkit_df)

# df = pd.read_csv('data/pki-ligands.csv')
# sns.set_style('darkgrid')
# sns.histplot(data=df, x='pKi', bins = 40, kde=True).set_title('Regression Dataset pKi distribution')
# plt.show()

# Y = [0, 0.25, 0.5, 0.75, 1]
# Y_axis = np.arange(len(Y))

# fig, ax = plt.subplots(1,1)
# ax.set_axisbelow(True)
# ax.set_facecolor('#ededed')
# svm_bars=ax.barh(Y_axis - 0.15, svm_validation_mccs, 0.1, label='SVM', color='red')
# mlp_bars=ax.barh(Y_axis - 0.05, mlp_validation_mccs, 0.1, label= 'MLP', color='blue')
# rf_bars=ax.barh(Y_axis + 0.05, rf_validation_mccs, 0.1, label= 'RF', color='#34eb49')
# ax.set_yticks(Y_axis,Y)
# ax.set_ylabel('Feature Threshold')
# ax.set_xlabel('MCC (%)')
# rects = ax.patches
# for rect in rects:
#     width = rect.get_width()
#     ax.text(
#         width+0.4, rect.get_y()+0.025, np.round(width,2)
#     )

# ax.legend(bbox_to_anchor=[1,0.57])
# ax.grid()
# ax.set_title('ECFP6 MCC Comparison')
# plt.show()
# all_validation_accs = [svm_validation_accs, mlp_validation_accs, rf_validation_accs]
# all_validation_f1s = [svm_validation_f1s, mlp_validation_f1s, rf_validation_f1s]
# all_validation_gmeans = [svm_validation_gmeans, mlp_validation_gmeans, rf_validation_gmeans]
# all_validation_mccs = [svm_validation_mccs, mlp_validation_mccs, rf_validation_mccs]

# for accs, f1s, gmeans, mccs, classifier in zip(all_validation_accs,all_validation_f1s, all_validation_gmeans, all_validation_mccs,['Support Vector Machine', 'Multilayer Perceptron', 'Random Forest']):
#     plot_validation_results(accs,f1s,gmeans,mccs,classifier)




# pki_df = pd.read_csv('pki-ligands.csv')
# pki_df = pki_df[['smiles', 'value']]

# ecfp6_df = ctk.computeDesc(list(pki_df['smiles']), list(pki_df['value']))
# rdkit_df = ctk.computeFP(list(pki_df['smiles']), list(pki_df['value']))
# ecfp6_df.dropna(inplace=True)
# rdkit_df.dropna(inplace=True)
# print(ecfp6_df)
# print(rdkit_df)
# ecfp6_df.to_csv('pki-ecfp6.csv', index=False)
# rdkit_df.to_csv('pki-rdkit.csv', index=False)