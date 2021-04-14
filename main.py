'''
Currently only looking at professional games, treating all sub-patches as identical
'''

# todo vary input parameters using forward or backward wrapper

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

# minimum number of games to count
min_games = 10

# todo temporary buff adjuster
buff_adjuster = 0.5

# create variable to investigate patches, where most recent patch is 1, two patches ago is 2, etc
# todo incorporate sub-patches using date ranges(?)
patch_ago = 2
url = 'https://www.datdota.com/heroes/performances?patch=7.28&after=01%2F01%2F2011&before=' \
      '12%2F04%2F2021&duration=' \
      '0%3B200&duration-value-from=0&duration-value-to=200&tier=2&valve-event=does-not-matter&threshold=1'
# use requests and bs to read the webpage as html txt file
example_file = requests.get(url)
print(example_file.raise_for_status())
soup = bs4.BeautifulSoup(example_file.text, 'html.parser')

# variables for algorithm
ban_list = ['1stphasepicks', '2ndphasepicks', '3rdphasepicks', '1stphasebans', '2ndphasebans', '3rdphasebans']

# weighted ave and stdev function
def weighted_ave_and_std(values, weights):
    '''
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape
    '''
    average = np.average(values, weights=weights)
    # Fast and numerically precise
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance))

def hero_cleanup(hero_df, total_games):
    # drop all columns with data based on other columns
    drop_columns = ['totalcount', 'winrate', 'kda', 'avgkal']
    hero_df = hero_df.drop(columns=drop_columns)

    # columns that should be normalized to number of games
    columns_to_game_normalize = ['wins', 'losses', 'asradiant', 'asdire'] + ban_list
    for i in columns_to_game_normalize:
        hero_df[i] = hero_df[i] / total_games
    return hero_df


def patch_grabber(patch_ago, min_games, soup):
    # grab url for professional DOTA matches
    patch = soup.select(f'#patch > option:nth-child({patch_ago})')
    patch = patch[0].getText()

    url_hero_stats = (f'https://www.datdota.com/heroes/performances?patch={patch}&after=01%2F01%2F2011&before='
                      '12%2F04%2F2021&duration='
                      '0%3B200&duration-value-from=0&duration-value-to=200&tier=2&valve-event=does-not-matter'
                      '&threshold=1')

    # use requests and bs to read the webpage as html txt file
    example_file = requests.get(url_hero_stats)
    print(example_file.raise_for_status())
    soup = bs4.BeautifulSoup(example_file.text, 'html.parser')

    # patch version to investigate
    patch = soup.select(f'#patch > option:nth-child({patch_ago})')
    patch = patch[0].getText()
    print(f'Current patch is {patch}')

    # locate the table with relevant information on heros during the patch in question
    hero_table = soup.find('table', class_='table table-striped table-bordered table-hover data-table')

    # import a string of punctuation and whitespace to be removed from strings
    exclist = string.punctuation + string.whitespace
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
                dota_row.append(dota_stat)
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
    url_bans = (
        'https://www.datdota.com/drafts?faction=both&first-pick=either&tier=2&valve-event=does-not-matter&patch='
        f'{patch}&after=01%2F01%2F2011&before=12%2F04%2F2021&duration=0%3B200&duration-value-from='
        '0&duration-value-to=200')

    # use requests and bs to read the webpage as html txt file
    ban_file = requests.get(url_bans)
    print(ban_file.raise_for_status())
    ban_soup = bs4.BeautifulSoup(ban_file.text, 'html.parser')

    # grab the number of total games
    total_games_soup = ban_soup.select('#page-wrapper > div.row.border-bottom.white-bg.dashboard-header > div > '
                                       'div.col-md-12 > div.table-responsive > h3')
    total_games = total_games_soup[0].getText()
    # create a list to eliminate letters and empty space
    exc_alpha = string.ascii_letters + string.whitespace + string.punctuation
    total_games = int(total_games.translate(str.maketrans('', '', exc_alpha)))
    print(f'Total number of games: {total_games}')

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
                dota_row.append(dota_stat)
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

    # transform the data into float
    hero_df = hero_df.astype('float')

    return hero_df, total_games


# %%

hero_df, total_games = patch_grabber(patch_ago, min_games, soup)

hero_df.to_csv('herodf.csv')

# %%
'''
The best way to tell if a hero was nerfed or buffed is to compare the win rate of the hero from the previous patch
to the current patch. We can use this methodology for all patches up to the most recent.
There is an issue in that all patches have sub-patches, such as patch 7.28 had 7.28a 7.28b and 7.28c. This will 
need to be clarified in future versions.
This methodology supposes that IF a nerf/buff happens, humans will be affected. I imagine that some buffs will go
unnoticed, as will some nerfs, but this method should be the most data driven. Instances like Ana's use of Io
that changed the meta in the final stages of a tournament, will likely be much harder to detect.
Also, it is possible for a hero to be buffed, but others around it are MORE buffed. I would treat this as a nerf, as
other heroes are stronger as a result of the patch. Relative performance is key.
Finally, there will be variance in every patch for hero winrate. If we want to determine whether a hero is buffed or
nerfed, we need to account for variance in hero performance. Arguably, the more games the hero has been played, the
more accurate that winrate is likely to be. Thus, instead of comparing the winrate of each hero to the average winrate
over that patch, we want to look at the normalized average and stdev. Conversely, we could also examine winrate
variance over previous patches for each individual hero, but this introduces massive uncertainty as each patch can
change hero mechanics.
'''
# previous patch version
previous_hero_df, previous_total_games = patch_grabber(patch_ago + 1, min_games, soup)

# previous_hero_df = hero_cleanup(previous_hero_df, previous_total_games)

previous_hero_df.to_csv('previousherodf.csv')

# todo previous_hero_df limited by min_games in its dataset, or hero_df dataset?
winrate_compare = (hero_df['winrate'][hero_df['totalcount'] >= min_games] -
                   previous_hero_df['winrate'][previous_hero_df['totalcount'] >= min_games]).dropna()
winrate_compare.to_csv('winratecompare.csv')
winrate_ave = winrate_compare.mean()
winrate_std = winrate_compare.std()
# print(np.array(winrate_compare))
# print(np.array(hero_df['totalcount'][hero_df['totalcount'] >= min_games]))
winrate_weighted_ave, winrate_weighted_std = (
    weighted_ave_and_std(np.array(winrate_compare),
                         np.array(hero_df['totalcount'][hero_df['totalcount'] >= min_games]))
)
winrate_high = winrate_weighted_ave + winrate_weighted_std*buff_adjuster
winrate_low = winrate_weighted_ave - winrate_weighted_std
buff_classifier = np.where(winrate_compare > winrate_high, 1, 0)
buff_classifier = pd.Series(buff_classifier, index=winrate_compare.index)
nerf_classifier = np.where(winrate_compare < winrate_low, 1, 0)
nerf_classifier = pd.Series(nerf_classifier, index=winrate_compare.index)

hero_df['winrate_compare'] = winrate_compare
hero_df['buffed'] = buff_classifier
hero_df['nerfed'] = nerf_classifier

hero_df.to_csv('herodf.csv')

# %%
'''
At this point, it is essential to separate the data so that our data is scaled only on data that falls within
the training set, rather than all data contained in the train, test, and validation splits.
'''
# eliminate all heroes that aren't classified, or have null values
hero_df = hero_df.dropna()

# separate the data
X = hero_df.iloc[:,:-3]
y = hero_df['buffed']

# do not touch X_val and y_val until you feel EXTREMELY CONFIDENT about your model
X_traintest, X_val, y_traintest, y_val = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

X_train, X_test, y_train, y_test = train_test_split(X_traintest, y_traintest, stratify=y_traintest)

# %%
# train a simple nearest neighbors classifier

# build a pipeline to standardize the input data and generate a model
# MinMaxScaler required as Chi2 only works for positive input values
pipe = Pipeline([('scaler', StandardScaler()),
                 ('minmax', MinMaxScaler()),
                 ('reduce_dim', SelectKBest(chi2)),
                 ('kNNClassifier', KNeighborsClassifier())])

# reduce_dim__k removes all but the highest scoring k features
param_grid = {
    'reduce_dim__k' : np.arange(1,15,1),
    'kNNClassifier__n_neighbors' : np.arange(2,11,1),
}

CV = GridSearchCV(pipe, param_grid, n_jobs=1, return_train_score=True, scoring='f1')

CV.fit(X_train, y_train)

cv_results = pd.DataFrame(CV.cv_results_)
cv_results.to_csv(f'CV_kNN_results.csv')

fig, ax = plt.subplots(1, 1, figsize=(12,9))
ax = sns.lineplot(data=cv_results, x='param_kNNClassifier__n_neighbors',
                     y='mean_test_score', hue='param_reduce_dim__k')
ax.set_title('5-fold CV performance for kNN Classifier')
ax.set_xlabel(f'k Nieghbors')
ax.legend(loc='upper right')
plt.savefig(f'kNN_Buff_score.png')
plt.clf()

report = classification_report(y_test, CV.predict(X_test), target_names=['not_buffed', 'buffed'], output_dict=True)
pd.DataFrame(report).T.to_csv('kNN_classification_report.csv')
print('Done with kNN')

# %%
# train a SVC

# build a pipeline to standardize the input data and generate a model
# MinMaxScaler required as Chi2 only works for positive input values
pipe = Pipeline([('scaler', StandardScaler()),
                 ('minmax', MinMaxScaler()),
                 ('reduce_dim', SelectKBest(chi2)),
                 ('classifier', NuSVC())])

param_grid = {
    'reduce_dim__k' : np.arange(1,15,1),
    'classifier__nu': np.linspace(0.01,1,11).tolist(),
    }  # 'classifier__kernel': ['linear', 'rbf']

CV = GridSearchCV(pipe, param_grid, n_jobs=1, return_train_score=True, scoring='f1')

CV.fit(X_train, y_train)

cv_results = pd.DataFrame(CV.cv_results_)
cv_results.to_csv(f'CV_NuSVC_results.csv')

fig, ax = plt.subplots(1, 1, figsize=(12,9))
ax = sns.lineplot(data=cv_results, x='param_classifier__nu',
                     y='mean_test_score', hue='param_reduce_dim__k')
ax.set_title('5-fold CV performance for NuSVC')
ax.set_xlabel(f'\u03BD for \u03BD-SVC')
ax.legend(loc='upper right')
plt.savefig(f'NuSVC_buff.png')
plt.clf()

report = classification_report(y_test, CV.predict(X_test), target_names=['not_buffed', 'buffed'], output_dict=True)
pd.DataFrame(report).T.to_csv('NuSVC_classification_report.csv')
print('done with SVC')

# %%
# train a RFC
'''
Current model has information leaking somehow. Training sets are all scoring perfectly, some unique info
is entering the model. Unclear what it is.
'''
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
# todo add logistic regression
