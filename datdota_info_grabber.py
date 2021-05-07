def datdota_info_grabber():
    # import streamlit as st
    from proxycreator import get_proxies
    import requests
    from bs4 import BeautifulSoup
    from itertools import cycle
    import string
    import pandas as pd
    import numpy as np
    import math
    import time
    import concurrent.futures as futures

    # minimum number of games to count
    min_games = 10

    # number of subpatches to analyze
    patch_ago = 50

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

    # todo add in what patch for print messages
    def patch_grabber(now_year, now_month, now_day, next_year, next_month, next_day):

        proxies = get_proxies()
        proxy_pool = cycle(proxies)
        n_proxies = 21

        for i in range(1, n_proxies):
            # Get a proxy from the pool
            proxy = next(proxy_pool)
            try:
                url_hero_stats = (f'https://www.datdota.com/heroes/performances?patch=7.29&patch=7.28&patch=7.27&patch=7.26'
                                  f'&patch=7.25&patch=7.24&patch=7.23&patch=7.22&patch=7.21&patch=7.20&patch=7.19&patch=7.18'
                                  f'&patch=7.17&patch=7.16&patch=7.15&patch=7.14&patch=7.13&patch=7.12&patch=7.11&patch=7.10'
                                  f'&patch=7.09&patch=7.08&patch=7.07&patch=7.06&patch=7.05&patch=7.04&patch=7.03&patch=7.02'
                                  f'&patch=7.01&patch=7.00&after={now_day}%2F{now_month}%2F{now_year}'
                                  f'&before={next_day}%2F{next_month}%2F{next_year}&duration=0%3B200'
                                  f'&duration-value-from=0&duration-value-to=200&tier=1&tier=2&tier=3'
                                  f'&valve-event=does-not-matter&threshold=1')

                # use requests and bs to read the webpage as html txt file
                example_file = requests.get(url_hero_stats, proxies={"http": proxy, "https": proxy})
                print('Subpatch hero pick info successfully scraped')
                # print(example_file.raise_for_status())
                soup = BeautifulSoup(example_file.text, 'html.parser')

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
                break
            except:
                print('Could not access subpatch hero pick info')
                pass

        # %%
        # Now we need to assemble information on picks and bans
        for i in range(1, n_proxies):
            # Get a proxy from the pool
            proxy = next(proxy_pool)
            try:
                url_bans = ('https://www.datdota.com/drafts?faction=both&first-pick=either&tier=1&tier=2&tier=3'
                            '&valve-event=does-not-matter&patch=7.29&patch=7.28&patch=7.27&patch=7.26&patch=7.25'
                            '&patch=7.24&patch=7.23&patch=7.22&patch=7.21&patch=7.20&patch=7.19&patch=7.18&patch=7.17'
                            '&patch=7.16&patch=7.15&patch=7.14&patch=7.13&patch=7.12&patch=7.11&patch=7.10&patch=7.09'
                            '&patch=7.08&patch=7.07&patch=7.06&patch=7.05&patch=7.04&patch=7.03&patch=7.02&patch=7.01'
                            f'&patch=7.00&after={now_day}%2F{now_month}%2F{now_year}'
                            f'&before={next_day}%2F{next_month}%2F{next_year}&duration=0%3B200&duration-value-from=0'
                            f'&duration-value-to=200')

                # use requests and bs to read the webpage as html txt file
                ban_file = requests.get(url_bans, proxies={"http": proxy, "https": proxy})
                print('Subpatch hero ban info successfully scraped')
                # print(ban_file.raise_for_status())
                ban_soup = BeautifulSoup(ban_file.text, 'html.parser')

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
                break
            except:
                print('Could not access subpatch hero ban info')
                pass

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

    # grab patch_time from local file
    patch_time = pd.read_csv('PatchTime.csv', header=0, index_col=0)

    hero_all_selected_patches = pd.DataFrame()
    for i in range(2, patch_ago + 1):
        print(patch_time.index[i])
        # obtain year, month, day for all relevent patches
        # st.write(f'Current patch is {patch_time.index[i]}, Next patch is {patch_time.index[i - 1]}')
        current_patch_year, current_patch_month, current_patch_day = patch_date_producer(i, patch_time)
        next_patch_year, next_patch_month, next_patch_day = patch_date_producer(i - 1, patch_time)
        next_next_patch_year, next_next_patch_month, next_next_patch_day = patch_date_producer(i - 2, patch_time)
        hero_df, total_games = patch_grabber(current_patch_year, current_patch_month, current_patch_day,
                                             next_patch_year, next_patch_month, next_patch_day)
        print('done with current patch')

        # %%
        next_hero_df, next_total_games = patch_grabber(next_patch_year, next_patch_month, next_patch_day,
                                                       next_next_patch_year, next_next_patch_month,
                                                       next_next_patch_day)
        print('done with following patch')

        # todo following_hero_df limited by min_games in its dataset, or hero_df dataset?
        winrate_compare = (hero_df['winrate'][hero_df['totalcount'] >= min_games] -
                           next_hero_df['winrate'][next_hero_df['totalcount'] >= min_games]).dropna()

        previous_hero_df_min_games = next_hero_df[next_hero_df['totalcount'] >= min_games]
        hero_df_min_games = hero_df[hero_df['totalcount'] >= min_games]

        # remove any hero not present in both patches
        hero_difference = hero_df_min_games.index.difference(previous_hero_df_min_games.index)
        hero_df_min_games = hero_df_min_games.drop(index=hero_difference)

        # winrate_weighted_ave, winrate_weighted_std = (
        #     weighted_ave_and_std(np.array(winrate_compare),
        #                          np.array(hero_df_min_games['totalcount']))
        # )
        # winrate_high = winrate_weighted_ave + winrate_weighted_std
        # winrate_low = winrate_weighted_ave - winrate_weighted_std
        # buff_classifier = np.where(winrate_compare > winrate_high, 1, 0)
        # buff_classifier = pd.Series(buff_classifier, index=winrate_compare.index)
        # nerf_classifier = np.where(winrate_compare < winrate_low, 1, 0)
        # nerf_classifier = pd.Series(nerf_classifier, index=winrate_compare.index)

        hero_df['winrate_compare'] = winrate_compare
        hero_df['patch'] = patch_time.index[i]
        # hero_df['buffed'] = buff_classifier
        # hero_df['nerfed'] = nerf_classifier

        hero_all_selected_patches = pd.concat([hero_all_selected_patches, hero_df])

    return hero_all_selected_patches

hero_all_subpatch_info = datdota_info_grabber()
hero_all_subpatch_info.to_csv(f'HeroAllPatchesTest.csv')