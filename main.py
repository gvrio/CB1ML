import pickle
import pandas as pd
import numpy as np
import statistics as sts
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from functools import partial
from hyperopt import hp, fmin, tpe, Trials, space_eval
from hyperopt.pyll.base import scope
from imblearn.metrics import geometric_mean_score
from imblearn.combine import SMOTETomek

STATE = 111

ensemble = {
    SVC: 
    {
        'C': hp.uniform('C', 0.1, 500)
    },
    MLPClassifier:
        {
            'hidden_layer_sizes': scope.int(hp.quniform('hidden_layer_sizes', 100, 1500, 100)),
            'activation': hp.choice('activation', ['tanh', 'relu']),
            'alpha': hp.uniform('alpha',0.0001, 1),
            'max_iters': scope.int(hp.quniform('max_iters', 300, 500))
        },
    RandomForestClassifier:   
    {
        'max_depth':scope.int(hp.quniform('max_depth', 3, 15, 1)),
        'n_estimators':scope.int(hp.quniform('n_estimators', 50, 1000, 1))
    }
    }

def load_data(file):
    df = pd.read_csv(file)
    X = df.drop('labels', axis=1)
    y = df['labels']
    features = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, stratify=y, random_state=STATE)
    return (X_train,X_test, y_train, y_test, features)

def resample_and_scale(X_train, X_test, y_train):  
    kmeans_smote = SMOTETomek(sampling_strategy='all', random_state=STATE)
    X_res, y_res = kmeans_smote.fit_resample(X_train, y_train)
    scaler = StandardScaler()
    X_res_scaled = scaler.fit_transform(X_res)
    X_test_scaled = scaler.fit_transform(X_test)
    return (X_res_scaled, X_test_scaled, y_res.to_numpy())

def rf_feature_selection(X, y, features):
    states = [111,222,333,444,555]
    importances = []
    accuracies = []
    for state in states:
        xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.7, shuffle=True, stratify=y, random_state=state)
        rf_feat_sel = RandomForestClassifier(random_state=STATE, oob_score=True)
        rf_feat_sel.fit(xtrain,ytrain)
        preds = rf_feat_sel.predict(xtest)
        mcc = metrics.matthews_corrcoef(ytest,preds)
        fold_mdis = rf_feat_sel.feature_importances_
        accuracies.append(mcc)
        importances.append(fold_mdis)
    importances_df = pd.DataFrame(importances, columns=features)
    importances_df['accuracies'] = accuracies
    return importances_df

def transform_features(feat_sel_res, X_train, X_test, threshold, features):
    best_acc_idx = feat_sel_res['accuracies'].idxmax()
    impurities =  dict(feat_sel_res.drop('accuracies', axis=1).iloc[best_acc_idx])
    median_mdi = sts.median(list(impurities.values()))
    best_features = [feat for feat in impurities if impurities[feat] > median_mdi * threshold]
    features_idx = [feat in best_features for feat in features]
    feat_sel_xtrain =  np.array([sample[features_idx] for sample in X_train])
    feat_sel_xtest = np.array([sample[features_idx] for sample in X_test])
    return (feat_sel_xtrain, feat_sel_xtest, best_features)

def optimize_model(params, X, y, estimator):
    model = estimator(**params, random_state=STATE)
    kfold = StratifiedKFold(n_splits=5, random_state=STATE, shuffle=True)
    accuracies =[]
    for idx in kfold.split(X=X, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain, xtest = X[train_idx], X[test_idx]
        ytrain, ytest = y[train_idx], y[test_idx]
        model.fit(xtrain,ytrain)
        preds = model.predict(xtest)
        acc = metrics.matthews_corrcoef(ytest,preds)
        accuracies.append(acc)
    return -1.0 * np.mean(accuracies)

def tree_parzen_optimization(X,y, model, params):
    optimize = partial(
        optimize_model,
        X=X,
        y=y,
        estimator=model
    )

    bayesian_params = fmin(
        fn=optimize,
        space=params,
        trials=Trials(),
        algo=tpe.suggest,
        max_evals=20
    )
    optimized_params = space_eval(params, bayesian_params)
    print(optimized_params)
    return optimized_params


def cross_validate_pipeline(estimator, X_train, y_train, params, features, th):
    cv_acc = []
    cv_f1 = []
    cv_gmean = []
    cv_mcc = []
    cv_params = []
    cv_features = []
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=STATE)
    for idx in kfold.split(X=X_train,y=y_train):
        
        train_idx, test_idx = idx[0], idx[1]
        xtrain, xtest = X_train[train_idx], X_train[test_idx]
        ytrain, ytest = y_train[train_idx], y_train[test_idx]
        
        feat_sel_xtrain, feat_sel_xtest, best_features = transform_features(
            feat_sel_res=rf_feature_selection(xtrain, ytrain, features),
            X_train=xtrain,
            X_test=xtest,
            features=features,
            threshold=th
        )
        
        
        best_params = tree_parzen_optimization(
            X=feat_sel_xtrain,
            y=ytrain,
            model=estimator,
            params=params
        )
        
        model = estimator(**best_params, random_state=STATE)
        model.fit(feat_sel_xtrain, ytrain)
        predictions = model.predict(feat_sel_xtest)
        
        acc = metrics.accuracy_score(ytest, predictions)
        f1 = metrics.f1_score(ytest, predictions)
        mcc = metrics.matthews_corrcoef(ytest, predictions)
        gmean = geometric_mean_score(ytest,predictions, average='binary')
        
        cv_acc.append(acc)
        cv_f1.append(f1)
        cv_mcc.append(mcc)
        cv_gmean.append(gmean)
        cv_params.append(best_params)
        cv_features.append(best_features)
    
    cv_res = dict()
    cv_res['acc'] = sts.mean(cv_acc)
    cv_res['f1'] = sts.mean(cv_f1)
    cv_res['mcc'] = sts.mean(cv_mcc)
    cv_res['gmean'] = sts.mean(cv_gmean)
    cv_res['params'] = cv_params[cv_mcc.index(max(cv_mcc))]
    cv_res['features'] = cv_features[cv_mcc.index(max(cv_mcc))]

    return cv_res

def train_split_validation(estimator, params, X_train, X_test, y_train, y_test):
    model = estimator(**params, random_state=STATE)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = metrics.accuracy_score(y_test, predictions)
    f1 = metrics.f1_score(y_test, predictions)
    mcc = metrics.matthews_corrcoef(y_test, predictions)
    gmean = geometric_mean_score(y_test,predictions, average='binary')
    c_matrix = metrics.confusion_matrix(y_test, predictions)
    tn, fp, fn, tp = c_matrix.ravel()
    
    results = dict()
    results['acc'] = acc
    results['f1'] = f1
    results['mcc'] = mcc
    results['gmean'] = gmean
    results['tn'] = tn
    results['fp'] = fp
    results['fn'] = fn
    results['tp'] = tp
    return results

def run_experiments(th):
    
    experimental_results = []
    X_train, X_test, y_train, y_test, features = load_data('rdkit-data.csv')
    X_train_scaled_res, X_test_scaled, y_train_res = resample_and_scale(X_train, X_test, y_train)
    
    for model in ensemble:
        params = ensemble[model]
        cv_results = cross_validate_pipeline(
            estimator=model, 
            X_train = X_train_scaled_res, 
            y_train=y_train_res, 
            params=params,
            features=features,
            th=th
            )
        best_params = cv_results['params']
        best_features = cv_results['features']
        features_idx = [feat in best_features for feat in features]
        feat_Xtrain =  np.array([sample[features_idx] for sample in X_train_scaled_res])
        feat_Xtest = np.array([sample[features_idx] for sample in X_test_scaled])
        
        validation_results = train_split_validation(
            estimator=model, 
            params=best_params, 
            X_train= feat_Xtrain, 
            X_test=feat_Xtest, 
            y_train=y_train_res,
            y_test=y_test
            )
        summary={'model': model, 'validation results': validation_results, 'cv results': cv_results}
        experimental_results.append(summary)
    return experimental_results

def benchmark_experiments(thresholds):
    for th in thresholds:
        results = run_experiments(th)
        output = f'''
        FEATURE THRESHOLD = Median(FeatureImpurity) x {th}
        -------------------------------------------------------
        RESULTS:\n\n
        {results}
        \n\n\n
        '''
        print(output)

def main():
    thresholds = [0, 0.25, 0.5, 0.75, 1]
    benchmark_experiments(thresholds)
        
    
if __name__ == '__main__':
    main()
