from hyperopt import hp
from hyperopt.pyll.base import scope
import numpy as np

params_space = {
    'rf':   {
        'max_depth':scope.int(hp.quniform('max_depth', 3, 15, 1)),
        'n_estimators':scope.int(hp.quniform('n_estimators', 50, 1000, 1)),
        'criterion':hp.choice('criterion', ['gini', 'entropy']),
        'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0.01, 1.0),
        'max_features': hp.uniform('max_features', 0.01, 1)
    },
    'svm': {
        'C': hp.uniform('C', 0.1, 1000),
        'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    },
    'mlp':
        {
            'hidden_layer_sizes': scope.int(hp.quniform('hidden_layer_sizes', 100, 1500, 100)),
            'activation': hp.choice('activate', ['identity', 'logistic', 'tanh', 'relu']),
            'alpha': hp.uniform('alpha',0.0001, 1)
        }
    }

opt_rf_params = {'criterion': 1, 'max_depth': 12, 'max_features': 0.06531127450867415, 'min_impurity_decrease': 0.685351085521038, 'n_estimators': 287}

arr = np.array([1,2,3,4,5])

print(arr[False])