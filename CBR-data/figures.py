import pickle
import numpy as np
import matplotlib.pyplot as plt


thresholds = [0, 0.25, 0.5, 0.75, 1]
def plot_validation_results(metrics, fig_name):
    X = [100, 87.5, 75.0, 62.5, 50.0]
    X.reverse()
    X_axis = np.arange(len(X))
    Y = list(range(0, 120, 20))
    Y_axis = np.arange(len(X))
    titles = ['Accuracy', 'F1-Score','Matthews Correlation Coef.']
    fig, axs = plt.subplots(3)
    fig.suptitle(fig_name, fontsize=18)
    
    for ax, metric, title in zip(axs.flat, metrics, titles):
        rdkit_metrics = metric['rdkit']
        ecfp6_metrics = metric['ecfp6']
        rdkit_metrics.reverse()
        ecfp6_metrics.reverse()
        ax.set_axisbelow(True)
        ax.set_facecolor('#ededed')
        ax.bar(X_axis - 0.035, rdkit_metrics, 0.07, label='RDKit', color='red')
        ax.bar(X_axis + 0.035, ecfp6_metrics, 0.07, label= 'ECFP6', color='blue')
        ax.set_xticks(X_axis,X)
        ax.set_ylim(top=115)
        ax.set_xlabel('% of features')
        ax.set_ylabel('Performance (%)')
        ax.label_outer()
        ax.set_title(title)
        rects = ax.patches
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x()+0.002, height + 2, np.round(height,1), fontsize='xx-small'
            )

        ax.grid()
    ax.legend()
    plt.show()
    
ecfp6_results = {}
for th in thresholds:
    file1 = f'results/ecfp6_class_results_{th}.pkl'
    file2 = f'results/rdkit_class_results_{th}.pkl'
    with open(file1, 'rb') as f1:
        res1 = pickle.load(f1)
        print(file1)
        print(res1[0]['validation results'])
        print(res1[1]['validation results'])
        print(res1[2]['validation results'])
    with open(file2, 'rb') as f2:
        res2 = pickle.load(f2)
        print(file2)
        print(res2[0]['validation results'])
        print(res2[1]['validation results'])
        print(res2[2]['validation results'])