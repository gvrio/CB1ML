import optuna
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from custom_funcs import computeFP

def build_model(learning_rate, num_hidden_layers, num_units, activation):
    model = keras.Sequential()
    model.add(layers.Flatten())
    for _ in range(num_hidden_layers):
        model.add(layers.Dense(num_units=num_units, activation=activation))
    model.add(keras.layers.Dense(1))
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
        loss='mean_squared_error', 
        metrics=['mean_squared_error']
        )
    
    return model

def objective(trial, x_train, y_train, x_val, y_val):
    learning_rate = trial.suggest_float('learning_rate', 0.001, 1)
    num_hidden_layers = trial.suggest_int('num_hidden_layers', 1, 4)
    num_units = trial.suggest_int('num_units', 64, 512)
    activation = trial.suggest_categorical('activation', ['relu', 'sigmoid', 'linear'])
    
    #build model
    model = build_model(learning_rate, num_hidden_layers, num_units, activation)
    
    model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=True)
    loss, mse = model.evaluate(x_val, y_val)
    
    return -mse

def optimized_model(params, losses):
    hp = params[losses.index(max(losses))]
    optmodel = build_model(hp['learning_rate'], hp['num_hidden_layers'], hp['num_units'], hp['activation'])
    return optmodel

def cross_validation(X, y):
    kf = KFold(shuffle=True, random_state=1)
    params = []
    losses = []

    for (train_idx, val_idx) in kf.split():
        xtrain, xval = x_train[train_idx], x_train[val_idx]
        ytrain, yval = y_train[train_idx], y_train[val_idx]
        
        study = optuna.create_study(sampler=optuna.samplers.HyperbandSampler(), direction='maximize')
        study.optimize(lambda trial: objective(trial, xtrain, ytrain, xval, yval), n_trials=50)
        best_params = study.best_params
        best_loss = study.best_value
        
        params.append(best_params)
        losses.append(best_loss)
        
    return (params, losses)

def fit_optmodel(params, losses, x_train, y_train, x_test, y_test):
    optmodel = optimized_model(params, losses)
    history = optmodel.fit(x_train, y_train, epochs=100, validation_split=0.2)
    return history
df = pd.read_csv('CBR-data/regression_data.csv')
fp_df = computeFP(list(df['smiles']), list(df['pki']))

X = fp_df.drop('labels', axis=1)
y = fp_df['labels']

x_train, y_train, x_test, y_test = train_test_split(X, y, train_size=0.7, shuffle=True, random_state=1)

print('Full set Statistics')
print('Size:', len(y))
print("Mean:", np.array(y).mean())
print("Standard Deviation:", np.array(y).std())
print("")

print("Train Set Statistics:")
print('Size:', len(y_train))
print("Mean:", np.array(y_train).mean())
print("Standard Deviation:", np.array(y_train).std())
print("")

print("Test Set Statistics:")
print('Size:', len(y_test))
print("Mean:", np.array(y_test).mean())
print("Standard Deviation:", np.array(y_test).std())



    