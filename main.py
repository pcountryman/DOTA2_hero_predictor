"""
Currently only looking at all semi-pro and above games, treating all sub-patches as identical
"""

import streamlit as st
import requests
import bs4
import string
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import NuSVC
from sklearn.metrics import classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow import keras

# which models to attempt to fit
knn = 'no'  # todo change to MLkNN or replace
svc = 'no'  # todo change to multilabel NuSVC or replace
ann = 'yes'

# minimum number of games to count
min_games = 10

# number of subpatches to analyze
patch_ago = 15

# todo temporary buff adjuster
buff_adjuster = 1
nerf_adjuster = 1

# create variable to investigate patches, where most recent patch is 1, two patches ago is 2, etc
patch_url = 'https://dota2.fandom.com/wiki/Game_Versions'
patch_file = requests.get(patch_url)
print(f'{patch_file.raise_for_status()} errors pulling patch versions')
patch_soup = bs4.BeautifulSoup(patch_file.text, 'html.parser')
# locate the table with relevant information on heros during the patch in question
patch_table = patch_soup.find('table', class_='wikitable')
# empty list to store column titles in
patch_table_columns = []

# import a string of punctuation and whitespace to be removed from strings
exclist = string.punctuation + string.whitespace
# go through each column title in the table and write it to dota_columns
for title in patch_table.find_all('th', class_='header'):
    column_title = title.text.strip()
    column_title = column_title.lower()

    # remove punctuation and whitespace from column_title
    column_title = column_title.translate(str.maketrans('', '', exclist))
    patch_table_columns.append(column_title)
patch_table_columns.pop(1)  # todo remove once highlights have been extracted

# empty dictionary to store all patches, their hero buff/nerfs, and their initialization dates
patch_dict = {}

# go through each row in the patch_table and write all stats to a dictionary. keys=patch, values=dates and heroes
for body in patch_table.find_all('tbody'):
    # todo fix once first row is fixed on website
    rows = body.find_all('tr')[2:]  # exclude first row as it is a header row
    for row in rows:
        patch_number = row.find('td').text.strip()
        # todo extract highlights based on a in li
        patch_date = row.find_all('td')[2].text.strip()
        patch_number = patch_number.translate(str.maketrans('', '', exclist))
        patch_dict[patch_number] = patch_date

# todo change to dataframe once highlights are included
# transform dictionary into a pandas series
patch_df = pd.Series(patch_dict, name='patch')
# convert into datetime info
patch_datetime = pd.Series(pd.to_datetime(patch_df, format='%Y/%m/%d'), name='datetime')
patch_year = patch_datetime.dt.year.rename('year')
patch_month = patch_datetime.dt.month.rename('month')
patch_day = patch_datetime.dt.day.rename('day')
patch_time = pd.concat([patch_year, patch_month, patch_day], axis=1)

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


def patch_grabber(now_year, now_month, now_day, next_year, next_month, next_day):
    url_hero_stats = (f'https://www.datdota.com/heroes/performances?patch=7.29&patch=7.28&patch=7.27&patch=7.26'
                      f'&patch=7.25&patch=7.24&patch=7.23&patch=7.22&patch=7.21&patch=7.20&patch=7.19&patch=7.18'
                      f'&patch=7.17&patch=7.16&patch=7.15&patch=7.14&patch=7.13&patch=7.12&patch=7.11&patch=7.10'
                      f'&patch=7.09&patch=7.08&patch=7.07&patch=7.06&patch=7.05&patch=7.04&patch=7.03&patch=7.02'
                      f'&patch=7.01&patch=7.00&after={now_day}%2F{now_month}%2F{now_year}'
                      f'&before={next_day}%2F{next_month}%2F{next_year}&duration=0%3B200'
                      f'&duration-value-from=0&duration-value-to=200&tier=1&tier=2&tier=3'
                      f'&valve-event=does-not-matter&threshold=1')

    # use requests and bs to read the webpage as html txt file
    example_file = requests.get(url_hero_stats)
    # print(example_file.raise_for_status())
    soup = bs4.BeautifulSoup(example_file.text, 'html.parser')

    # locate the table with relevant information on heros during the patch in question
    hero_table = soup.find('table', class_='table table-striped table-bordered table-hover data-table')

    # import a string of punctuation and whitespace to be removed from strings
    exclist = string.punctuation + string.whitespace
    punc_no_period = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~' + string.whitespace
    table = str.maketrans(dict.fromkeys(punc_no_period))
    # empty list to store column titles in
    dota_columns = []

    # go through each column title in the table and write it to dota_columns
    for column in hero_table.find_all('thead'):
        rows = column.find_all('tr')
        for row in rows:
            titles = row.find_all('th')
            for title in titles:
                column_title = title.text.strip()
                column_title = column_title.lower()

                # remove punctuation and whitespace from column_title
                column_title = column_title.translate(str.maketrans('', '', exclist))
                dota_columns.append(column_title)

    # empty dictionary to store all hero statistics
    hero_dict = {}

    # go through each row in the hero_table and write all stats to a dictionary. keys=hero_name, values=all hero stats
    for hero in hero_table.find_all('tbody'):
        rows = hero.find_all('tr')
        for row in rows:
            dota_row = []
            for i in range(1, len(row.find_all('td'))):
                dota_stat = row.find_all('td')[i].text.strip()
                dota_stat = dota_stat.rstrip('%')
                clean_dota_stat = dota_stat.translate(table)
                if clean_dota_stat == '':
                    clean_dota_stat = None
                dota_row.append(clean_dota_stat)
            hero_name = row.find('td').text.strip()
            hero_name = hero_name.translate(str.maketrans('', '', exclist))
            hero_dict[hero_name] = dota_row

    # transform dictionary into a pandas dataframe
    hero_df = pd.DataFrame(hero_dict).T

    # create a renaming dictionary because the column labels are currently 0, 1, etc
    rename_dict = {}
    for i in range(0, len(hero_df.columns)):
        rename_dict[hero_df.columns[i]] = dota_columns[i + 1]
    hero_df = hero_df.rename(columns=rename_dict)

    # %%
    # Now we need to assemble information on picks and bans

    url_bans = ('https://www.datdota.com/drafts?faction=both&first-pick=either&tier=1&tier=2&tier=3'
                '&valve-event=does-not-matter&patch=7.29&patch=7.28&patch=7.27&patch=7.26&patch=7.25'
                '&patch=7.24&patch=7.23&patch=7.22&patch=7.21&patch=7.20&patch=7.19&patch=7.18&patch=7.17'
                '&patch=7.16&patch=7.15&patch=7.14&patch=7.13&patch=7.12&patch=7.11&patch=7.10&patch=7.09'
                '&patch=7.08&patch=7.07&patch=7.06&patch=7.05&patch=7.04&patch=7.03&patch=7.02&patch=7.01'
                f'&patch=7.00&after={now_day}%2F{now_month}%2F{now_year}'
                f'&before={next_day}%2F{next_month}%2F{next_year}&duration=0%3B200&duration-value-from=0'
                f'&duration-value-to=200')

    # use requests and bs to read the webpage as html txt file
    ban_file = requests.get(url_bans)
    # print(ban_file.raise_for_status())
    ban_soup = bs4.BeautifulSoup(ban_file.text, 'html.parser')

    # grab the number of total games
    total_games_soup = ban_soup.select('#page-wrapper > div.row.border-bottom.white-bg.dashboard-header > div > '
                                       'div.col-md-12 > div.table-responsive > h3')
    total_games = total_games_soup[0].getText()
    # create a list to eliminate letters and empty space
    exc_alpha = string.ascii_letters + string.whitespace + string.punctuation
    total_games = int(total_games.translate(str.maketrans('', '', exc_alpha)))
    # print(f'Total number of games: {total_games}')

    # locate the table with relevant information on heros during the patch in question
    ban_table = ban_soup.find('table', class_='table table-striped table-bordered table-hover data-table')

    # empty list to store column titles in
    ban_columns = []

    # go through each column title in the table and write it to dota_columns
    for column in ban_table.find_all('thead'):
        row = column.find('tr')
        picks_and_bans = row.find_all('th')[1:3]
        for p_b in picks_and_bans:
            pick_or_ban = p_b.text
            table_row = column.find_all('tr')[1]
            if pick_or_ban == 'Picks':  # find a more elegant way to do this
                titles = table_row.find_all('th')[0:7]
            else:
                titles = table_row.find_all('th')[7:13]
            for title in titles:
                column_title = (title.text + pick_or_ban).strip()
                column_title = column_title.lower()

                # remove punctuation and whitespace from column_title
                column_title = column_title.translate(str.maketrans('', '', exclist))
                ban_columns.append(column_title)

    # empty dictionary to store all hero statistics
    ban_dict = {}

    # go through each row in the hero_table and write all stats to a dictionary. keys=hero_name, values=all hero stats
    for ban in ban_table.find_all('tbody'):
        rows = ban.find_all('tr')
        for row in rows:
            dota_row = []
            for i in range(1, len(row.find_all('td'))):
                dota_stat = row.find_all('td')[i].text.strip()
                clean_dota_stat = dota_stat.translate(table)
                if clean_dota_stat == '':
                    clean_dota_stat = None
                dota_row.append(clean_dota_stat)
            hero_name = row.find('td').text.strip()
            hero_name = hero_name.translate(str.maketrans('', '', exclist))
            ban_dict[hero_name] = dota_row

    # transform dictionary into a pandas dataframe
    ban_df = pd.DataFrame(ban_dict).T

    # create a renaming dictionary because the column labels are currently 0, 1, etc
    rename_dict = {}
    for i in range(0, len(ban_df.columns[1:])):
        rename_dict[ban_df.columns[i]] = ban_columns[i + 1]
    ban_df = ban_df.rename(columns=rename_dict)

    # aggregate pick/ban info into main hero_df
    for i in ban_list:
        hero_df[i] = ban_df[i]

    # transform the data into float and remove rows with empty data entries
    # hero_df.to_csv(f'herodfTEST.csv')
    hero_df = hero_df.dropna(axis=0)
    hero_df = hero_df.astype('float')

    return hero_df, total_games


# %%
'''
Scan over all patches up to patch_ago to capture more data.
'''


# create definition to fetch year, month, and date for a given patch
def patch_date_producer(i, patch_time):
    now_patch_year = patch_time['year'].iloc[i]
    now_patch_month = patch_time['month'].iloc[i]
    now_patch_day = patch_time['day'].iloc[i]
    return now_patch_year, now_patch_month, now_patch_day


hero_all_selected_patches = pd.DataFrame()
for i in range(2, patch_ago + 1):
    # obtain year, month, day for all relevent patches
    st.write(f'Current patch is {patch_time.index[i]}, Next patch is {patch_time.index[i - 1]}')
    current_patch_year, current_patch_month, current_patch_day = patch_date_producer(i, patch_time)
    next_patch_year, next_patch_month, next_patch_day = patch_date_producer(i - 1, patch_time)
    next_next_patch_year, next_next_patch_month, next_next_patch_day = patch_date_producer(i - 2, patch_time)
    hero_df, total_games = patch_grabber(current_patch_year, current_patch_month, current_patch_day,
                                         next_patch_year, next_patch_month, next_patch_day)

    # %%
    '''
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
    '''
    # next patch version
    next_hero_df, next_total_games = patch_grabber(next_patch_year, next_patch_month, next_patch_day,
                                                   next_next_patch_year, next_next_patch_month,
                                                   next_next_patch_day)

    # todo following_hero_df limited by min_games in its dataset, or hero_df dataset?
    winrate_compare = (hero_df['winrate'][hero_df['totalcount'] >= min_games] -
                       next_hero_df['winrate'][next_hero_df['totalcount'] >= min_games]).dropna()

    previous_hero_df_min_games = next_hero_df[next_hero_df['totalcount'] >= min_games]
    hero_df_min_games = hero_df[hero_df['totalcount'] >= min_games]

    # remove any hero not present in both patches
    hero_difference = hero_df_min_games.index.difference(previous_hero_df_min_games.index)
    hero_df_min_games = hero_df_min_games.drop(index=hero_difference)

    winrate_weighted_ave, winrate_weighted_std = (
        weighted_ave_and_std(np.array(winrate_compare),
                             np.array(hero_df_min_games['totalcount']))
    )
    winrate_high = winrate_weighted_ave + winrate_weighted_std * buff_adjuster
    winrate_low = winrate_weighted_ave - winrate_weighted_std * nerf_adjuster
    buff_classifier = np.where(winrate_compare > winrate_high, 1, 0)
    buff_classifier = pd.Series(buff_classifier, index=winrate_compare.index)
    nerf_classifier = np.where(winrate_compare < winrate_low, 1, 0)
    nerf_classifier = pd.Series(nerf_classifier, index=winrate_compare.index)

    hero_df['winrate_compare'] = winrate_compare
    hero_df['buffed'] = buff_classifier
    hero_df['nerfed'] = nerf_classifier

    hero_all_selected_patches = pd.concat([hero_all_selected_patches, hero_df])

hero_all_selected_patches.to_csv(f'heroallpatchesTEST.csv')

# %%
'''
At this point, it is essential to separate the data so that our data is scaled only on data that falls within
the training set, rather than all data contained in the train, test, and validation splits.
'''
# eliminate all heroes that aren't classified, or have null values
hero_all_selected_patches = hero_all_selected_patches.dropna()

# separate the data
X = hero_all_selected_patches.iloc[:, :-3]
y = hero_all_selected_patches[['buffed', 'nerfed']]

# do not touch X_val and y_val until you feel EXTREMELY CONFIDENT about your model
X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, stratify=y_full_train)

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

'''
Consider how obvious it is for humans to tell if a hero will be buffed or nerfed. Reddit sentiment analysis
all of the whiners, how accurate are they vs the experts.
Patch note based analysis to predict hero winrate.
Players over 6k MMR.
'''

# train a neural network
if ann == 'yes':
    '''
    Can change number of hidden layers, neurons per layer, learning rate, 
    number of input layers, architecture of layers, activation function, 
    weight initialization logic, etc
    '''


    def build_classifier_model(n_hidden=3, n_neurons=100, learning_rate=3e-3, input_shape=[26]):
        model = keras.models.Sequential()
        model.add(keras.layers.InputLayer(input_shape=input_shape))
        for layer in range(n_hidden):
            model.add(keras.layers.Dense(n_neurons, activation='relu'))
        model.add(keras.layers.Dense(2, activation='sigmoid'))
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])
        return model


    # build a classifier model using build_model
    model = build_classifier_model()

    history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)])

    # plot the results
    # todo label axis, bro
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)  # set the vertical range between 0,1
    plt.title('ANN predicts DOTA2 hero buffs and nerfs')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.savefig(f'Sequential_Neural_Net.png')
    print('done with Neural Network')

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
