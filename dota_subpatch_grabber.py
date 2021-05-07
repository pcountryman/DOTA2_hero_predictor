def dota_subpatch_grabber():
    # import streamlit as st
    from itertools import cycle
    from proxycreator import get_proxies
    import requests
    from bs4 import BeautifulSoup
    import string
    import pandas as pd

    proxies = get_proxies()
    proxy_pool = cycle(proxies)
    n_proxies = 21

    for i in range(1, n_proxies):
        # Get a proxy from the pool
        proxy = next(proxy_pool)
        try:
            patch_url = 'https://dota2.fandom.com/wiki/Game_Versions'
            patch_file = requests.get(patch_url, proxies={"http": proxy, "https": proxy})
            print('Patch info retrieved')
            # print(f'{patch_file.raise_for_status()} errors pulling patch versions')
            patch_soup = BeautifulSoup(patch_file.text, 'html.parser')
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

            patch_df.to_csv(f'PatchDfTest.csv')
            patch_time.to_csv(f'PatchTime.csv')
            break
        except:
            print('Could not access subpatch info')
            pass

dota_subpatch_grabber()