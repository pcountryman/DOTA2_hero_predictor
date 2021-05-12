"""
Currently only looking at all semi-pro and above games.
"""
"""
The best way to tell if a hero WILL BE nerfed or buffed is to compare the win rate of the hero from the current 
patch to the following patch. We can use this methodology for all patches up to the most recent.
There is an issue in that all patches have sub-patches, such as patch 7.28 had 7.28a 7.28b and 7.28c. This will 
need to be clarified in future versions.
This methodology supposes that IF a nerf/buff happens, humans will be affected. I imagine that some buffs will go
unnoticed, as will some nerfs, but this method should be the most data driven. Instances like Ana's use of Io
that changed the meta in the final stages of a tournament, will likely be much harder to detect.
Also, it is possible for a hero to be buffed, but others around it are MORE buffed. I would treat this as a nerf, as
other heroes are stronger as a result of the patch. Relative performance is key.
Finally, there will be variance in every patch for hero winrate. If we want to determine whether a hero is buffed or
nerfed, we need to account for variance in hero performance. Arguably, the more games the hero has been played, the
more accurate that winrate is likely to be. Thus, instead of comparing the winrate of each hero to the average 
winrate over that patch, we want to look at the normalized average and stdev. Conversely, we could also examine 
winrate variance over previous patches for each individual hero, but this introduces massive uncertainty as each 
patch can change hero mechanics.
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup
import string
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from tensorflow import keras

# which models to attempt to fit
knn = 'no'  # todo change to MLkNN or replace
svc = 'no'  # todo change to multilabel NuSVC or replace
ann = 'yes'

# minimum number of games to count
min_games = 10

# number of subpatches to analyze
patch_ago = 2

# variables for algorithm
ban_list = ['1stphasepicks', '2ndphasepicks', '3rdphasepicks', '1stphasebans', '2ndphasebans', '3rdphasebans']

# weighted ave and stdev function
def weighted_ave_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise
    variance = np.average((values - average) ** 2, weights=weights)
    return average, math.sqrt(variance)


def hero_cleanup(hero_df, total_games):
    # drop all columns with data based on other columns
    drop_columns = ['totalcount', 'winrate', 'kda', 'avgkal']
    hero_df = hero_df.drop(columns=drop_columns)

    # columns that should be normalized to number of games
    columns_to_game_normalize = ['wins', 'losses', 'asradiant', 'asdire'] + ban_list
    for j in columns_to_game_normalize:
        hero_df[j] = hero_df[j] / total_games
    return hero_df

# eliminate all heroes that aren't classified, or have null values
hero_all_selected_patches = pd.read_csv('HeroAllPatches.csv').dropna()

# separate the data
X = hero_all_selected_patches.iloc[:, 1:-2]
y = hero_all_selected_patches['winrate_compare']

# do not touch X_val and y_val until you feel EXTREMELY CONFIDENT about your model
X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# set seed for validation creation to be used later in history reconstruction for ANN history
seed = random.seed()
X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, random_state=seed)

# %%
# train a simple nearest neighbors classifier
if knn == 'yes':
    # build a pipeline to standardize the input data and generate a model
    # MinMaxScaler required as Chi2 only works for positive input values
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('minmax', MinMaxScaler()),
                     ('reduce_dim', SelectKBest(chi2)),
                     ('kNNClassifier', KNeighborsClassifier())])

    # reduce_dim__k removes all but the highest scoring k features
    param_grid = {
        'reduce_dim__k': np.arange(1, 15, 1),
        'kNNClassifier__n_neighbors': np.arange(2, 11, 1),
    }

    CV = GridSearchCV(pipe, param_grid, n_jobs=1, return_train_score=True, scoring='f1')

    CV.fit(X_train, y_train)

    cv_results = pd.DataFrame(CV.cv_results_)
    cv_results.to_csv(f'CV_kNN_results.csv')

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax = sns.lineplot(data=cv_results, x='param_kNNClassifier__n_neighbors',
                      y='mean_test_score', hue='param_reduce_dim__k')
    ax.set_title('5-fold CV performance for kNN Classifier')
    ax.set_xlabel(f'k Nieghbors')
    ax.legend(loc='upper right')
    plt.savefig(f'kNN_Buff_score.png')
    plt.clf()

    report = classification_report(y_val, CV.predict(X_val), target_names=['not_buffed', 'buffed'], output_dict=True)
    pd.DataFrame(report).T.to_csv('kNN_classification_report.csv')
    print('Done with kNN')

# %%
# train a SVC
if svc == 'yes':
    # build a pipeline to standardize the input data and generate a model
    # MinMaxScaler required as Chi2 only works for positive input values
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('minmax', MinMaxScaler()),
                     ('reduce_dim', SelectKBest(chi2)),
                     ('classifier', NuSVC())])

    param_grid = {
        'reduce_dim__k': np.arange(1, 15, 1),
        'classifier__nu': np.linspace(0.01, 1, 11).tolist(),
    }  # 'classifier__kernel': ['linear', 'rbf']

    CV = GridSearchCV(pipe, param_grid, n_jobs=1, return_train_score=True, scoring='f1')

    CV.fit(X_train, y_train)

    cv_results = pd.DataFrame(CV.cv_results_)
    cv_results.to_csv(f'CV_NuSVC_results.csv')

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ax = sns.lineplot(data=cv_results, x='param_classifier__nu',
                      y='mean_test_score', hue='param_reduce_dim__k')
    ax.set_title('5-fold CV performance for NuSVC')
    ax.set_xlabel(f'\u03BD for \u03BD-SVC')
    ax.legend(loc='upper right')
    plt.savefig(f'NuSVC_buff.png')
    plt.clf()

    report = classification_report(y_val, CV.predict(X_val), target_names=['not_buffed', 'buffed'], output_dict=True)
    pd.DataFrame(report).T.to_csv('NuSVC_classification_report.csv')
    print('done with SVC')

# %%

# todo add in CV
# todo output confidence percent for each prediction

"""
Consider how obvious it is for humans to tell if a hero will be buffed or nerfed. Reddit sentiment analysis
all of the whiners, how accurate are they vs the experts.
Patch note based analysis to predict hero winrate.
Players over 6k MMR.
"""

# train a neural network
if ann == 'yes':
    '''
    Can change number of hidden layers, neurons per layer, learning rate, 
    number of input layers, architecture of layers, activation function, 
    weight initialization logic, etc
    '''

    # scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    pd.DataFrame(X_train, columns=X.columns).to_csv(f'Xtrainscaled.csv')


    def build_regressor_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[26]):
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation='relu'))
        model.add(keras.layers.Dense(1))
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=[keras.metrics.RootMeanSquaredError()])
        return model


    keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_regressor_model)

    param_distribs = {
        'n_hidden': (1, 5, 10, 15, 20, 25, 30),
        'n_neurons': (25, 50, 75, 100, 200, 300),
        'learning_rate': (0.0003, 0.003, 0.03, 0.3),
        # 'batch_size': (5, 10, 15, 20, 25, 30, 35, 40)
    }

    rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=5, cv=3, return_train_score=True)

    rnd_search_fit = rnd_search_cv.fit(X_train, y_train, epochs=70, validation_data=(X_val_scaled, y_val),
                                       callbacks=[keras.callbacks.EarlyStopping(
                                           monitor='val_loss', patience=30)])

    model_predict = rnd_search_cv.predict(X_val_scaled)

    st.text(f'Best parameters are {rnd_search_cv.best_params_}')
    st.text(f'Best score is {rnd_search_cv.best_score_}')

    st.dataframe(rnd_search_cv.cv_results_)

    # take best results and generate history
    params = rnd_search_cv.best_params_

    model = build_regressor_model(n_hidden=params['n_hidden'], n_neurons=params['n_neurons'],
                                  learning_rate=params['learning_rate'])

    history = model.fit(X_train, y_train, epochs=200, validation_data=(X_val_scaled, y_val),
                        callbacks=[keras.callbacks.EarlyStopping(
                            monitor='val_loss', patience=30)])

    # plot the results
    fig, ax = plt.subplots(figsize=(8, 5))
    pd.DataFrame(history.history).plot(ax=ax)
    plt.grid(True)
    # plt.gca().set_ylim(0, 1)  # set the vertical range between 0,1
    plt.title('ANN predicts DOTA2 hero buffs and nerfs')
    plt.ylabel('MSE')
    plt.xlabel('epoch')
    plt.savefig(f'Sequential_Neural_Net.png')
    st.pyplot(fig)
    st.dataframe(hero_all_selected_patches)
    val_df = pd.DataFrame(model_predict, index=X_val.index)
    val_df['true_winrate_compare'] = y_val
    st.dataframe(val_df)
    st.text('done with Neural Network')

# %%

# train a RFC
'''
Current model has information leaking somehow. Training sets are all scoring perfectly, some unique info
is entering the model. Unclear what it is.

# build a pipeline to standardize the input data and generate a model
# MinMaxScaler required as Chi2 only works for positive input values
pipe = Pipeline([('scaler', StandardScaler()),
                 ('minmax', MinMaxScaler()),
                 ('reduce_dim', SelectKBest(chi2)),
                 ('classifier', RandomForestClassifier())])

param_grid = {
    'reduce_dim__k' : np.arange(1,15,1),
    'classifier__n_estimators': [200, 500],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth': [2, 3, 4, 5, 6, 7],
    'classifier__criterion': ['gini', 'entropy']}  # entropy was underperforming

CV = GridSearchCV(pipe, param_grid, n_jobs=1, return_train_score=True, scoring='f1')

CV.fit(X_train, y_train)

cv_results = pd.DataFrame(CV.cv_results_)
cv_results.to_csv(f'CV_RFC_results.csv')

fig, ax = plt.subplots(1, 1, figsize=(12,9))
ax = sns.scatterplot(data=cv_results, x='param_classifier__max_depth',
                     y='mean_test_score', hue='param_reduce_dim__k',
                     )  # style='param_classifier__max_features'
ax.set_title('5-fold CV performance for Random Forest Classifier')
ax.set_xlabel(f'Max Depth')
ax.legend(loc='upper right')
plt.savefig(f'RFC_buff.png')
plt.clf()

report = classification_report(y_test, CV.predict(X_test), target_names=['not_buffed', 'buffed'], output_dict=True)
pd.DataFrame(report).T.to_csv('RFC_classification_report.csv')
print('done with RFC')
'''

# todo add logistic regression
