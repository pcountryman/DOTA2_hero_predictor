'''
Read patch notes and make estimates of hero win rate adjustment accordingly.
Not best use for neural nets as there isn't a lot of data on patches (limited to number of patches)
'''

import requests
from bs4 import BeautifulSoup
import string
import pandas as pd

# create variable to investigate patches, where most recent patch is 1, two patches ago is 2, etc
patch_url = 'https://dota2.fandom.com/wiki/Version_7.29b'
patch_file = requests.get(patch_url)
# print(f'{patch_file.raise_for_status()} errors pulling patch versions')
patch_soup = BeautifulSoup(patch_file.text, 'html.parser')

hero_list = []

for div3 in patch_soup.find_all('div', class_='mw-parser-output'):
    for b in div3.find_all('b'):
        for a in b.find_all('a'):
            print(a.text.strip())
    for abilities in div3.find_all('ul'):
        for ability in abilities.find_all('li'):
            # for a in ability.find_all('a'):
            #     print(a.text.strip())
            print(ability.text.strip())



# mw-content-text
'''
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
'''