import pandas as pd
import numpy as np
from prediction import (no_sales_criteria, quantile_range, three_sigma_borders, best_params_founder, SimpleSmooth_Seas,
                        Holt_Seas, Holt_Winters, Arima, best_models_fit, define_best_model_test_np)
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from etna.datasets.tsdataset import TSDataset
from etna.models.prophet import ProphetModel
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statistics
from pathlib import Path
import pyodbc
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random

np.random.seed(42)  # Фиксация seed для numpy
random.seed(42)  # Фиксация seed для стандартной библиотеки Python


def metadata_DB(chain_list,  # List of necessary buyers
                category_list,  # List of necessary categories of goods
                status_name):  # Type of sales (if status_name=0, we will download all type of sales)
    CONNECTION_PATH = Path()
    # Connection parameters are inside txt file
    FILENAME = "connection_Lactalis.txt"
    CONNECTION_FILENAME = CONNECTION_PATH / FILENAME
    with open(CONNECTION_FILENAME) as f:
        lines = f.readlines()
    connect_str = ""
    # database_name = lines[2].split("=")[1][:-2]
    for x in lines:
        connect_str += x.replace('/n', '')
    connect_str = " ".join(connect_str.split())

    connection = pyodbc.connect(connect_str)

    # Extracting full list of buyers and categories of goods in case when we will predict full data
    # chain_fulllist = pd.read_sql(f"SELECT [id] FROM [dbo].[_spr_address_cpg]",
    #                              connection)
    # category_fulllist = pd.read_sql(f"SELECT [id] FROM [dbo].[_spr_sku_ppg]",
    #                                 connection)
    chain_fulllist = pd.read_sql(f'SELECT "id" FROM public."_spr_address_cpg";',
                                 connection)
    category_fulllist = pd.read_sql(f'SELECT "id" FROM public."_spr_sku_ppg";',
                                    connection)
    if len(chain_list) < 1:
        chain_list = chain_fulllist.id.to_list()
    if len(category_list) < 1:
        category_list = category_fulllist.id.to_list()

    chain_str = '('
    for x in chain_list:
        chain_str += str(x) + ','
    chain_str = chain_str[:-1] + ')'

    category_str = '('
    for x in category_list:
        category_str += str(x) + ','
    category_str = category_str[:-1] + ')'

    # Extracting data from SQL database
    if status_name == 0:
        # data = pd.read_sql(f"SELECT * FROM [dbo].[SalesInWeek] "
        #                    f"WHERE cpg_id in {chain_str} and ppg_id in {category_str};",
        #                    connection)
        data = pd.read_sql(f'SELECT * FROM SalesInWeek '
                           f'WHERE cpg_id in {chain_str} and ppg_id in {category_str} and status_id in (1,2);',
                           connection)
    elif status_name in (1, 2):
        # data = pd.read_sql(f"SELECT * FROM [dbo].[SalesInWeek] "
        #                    f"WHERE cpg_id in {chain_str} and ppg_id in {category_str} "
        #                    f"and status_id = {status_name}",
        #                    connection)
        data = pd.read_sql(f'SELECT * FROM SalesInWeek '
                           f'WHERE cpg_id in {chain_str} and ppg_id in {category_str} and status_id = {status_name};',
                           connection)
    else:
        print("Incorrect status_id")
        data = pd.DataFrame()

    print(f"Was extracted {len(data)} string")

    # Extracting table with years and id
    # year_calendar = pd.read_sql("SELECT [id], [year] FROM [MVPPresident].[dbo].[_spr_date_year]",
    #                             connection)
    year_calendar = pd.read_sql('SELECT "id", "year" FROM public."_spr_date_year"',
                                connection)

    day_of_week = pd.read_sql('SELECT * FROM public."_spr_split_day"',
                              connection)
    day_of_week['day_of_week'] = day_of_week['day'] - 1
    day_of_week = day_of_week.rename(columns={'proportion': 'Percent_dw'})

    year_calendar = year_calendar.astype({'id': np.int64, 'year': np.int64})

    connection.close()

    return data, year_calendar, day_of_week[['day_of_week', 'Percent_dw']]


# Build-up Calendar DataFrame (year, week_id) between start and end points
def sample_calendar_week(start_year: int,
                         start_week: int,
                         final_year: int,
                         final_week: int):
    res = []
    s = 1
    f = 52
    for y in range(start_year, final_year + 1):
        if y == start_year:
            s = start_week
        elif y == final_year:
            f = final_week
        for x in range(s, f + 1):
            res.append([str(y) + '_' + str(x)])
            s = 1
            f = 52
    calendar = pd.DataFrame(res)
    return calendar


def get_date_from_year_week(year, week):
    first_day_of_year = datetime(int(year), 1, 1)
    if year == 2023:
        days_to_add = timedelta(weeks=int(week))
    elif year == 2024:
        days_to_add = timedelta(weeks=int(week) - 1)
    else:
        days_to_add = timedelta(weeks=int(week) - 1)
    start_of_week = first_day_of_year + days_to_add - timedelta(days=first_day_of_year.weekday())
    return start_of_week.date()


# Developing of prediction loop for weeks dataset (in process now)
def prediction_week_month_approach(df, coef_smoothing, filling_calendar='yes', start_period=0,
                                   end_period=datetime.now()):
    pd.set_option('display.max_columns', None, 'display.width', None, 'display.max_rows', None)
    # l3_list = df.l3_id.value_counts().index.to_list()
    chain_list = df.cpg_id.value_counts().index.to_list()
    print(chain_list)
    chain_list = [697]

    for chain in chain_list:
        print("Prediction ", chain)
        df_final_full = pd.DataFrame(columns=['l3_id', 'buyer_id', 'year_id', 'month_id', 'volume',
                                              'predict_smoothing_seas', 'predict_holt_seas',
                                              'predict_arima', 'predict_holt_wint', 'sellin_corr',
                                              'predict_smoothing_corr_seas', 'predict_holt_corr_seas',
                                              'predict_arima_corr', 'predict_holt_wint_corr',
                                              'date_upload', 'best_model_total', 'best_model', 'status_id',
                                              'best_model_value'])

        l3_list = df[df.cpg_id == chain].l3_id.value_counts().index.to_list()
        print(l3_list)
        l3_list = [17, 18]

        for l3 in l3_list:

            print('Prediction', chain, 'group ', l3)
            df_1 = df[(df.l3_id == l3) & (df.cpg_id == chain)]
            df_1 = df_1[['l3_id', 'l3_name', 'year', 'week', 'volume', 'cpg_id', 'cpg_name']]
            df_1 = df_1.groupby(by=['l3_id', 'l3_name', 'year', 'week', 'cpg_id', 'cpg_name'],
                                as_index=False).sum()
            df_1['Cal'] = df_1.apply(lambda var: str(int(var.year)) + '_' + str(int(var.week)), axis=1)
            # df_1 = df_1[df_1.status_id == 2]
            # print(df_1)
            print(df_1.volume.sum())

            calendar = sample_calendar_week(df_1.year.min(),
                                            df_1[df_1.year == df_1.year.min()].week.min(),
                                            2024,
                                            6)

            # def
            if filling_calendar == 'yes':
                df_1 = df_1.merge(calendar, left_on='Cal', right_on=0, how='right')
                df_1['volume'] = df_1['volume'].fillna(0)
                df_1['year'] = df_1.apply(lambda var: int(var.Cal.split('_')[0]), axis=1)
                df_1['week'] = df_1.apply(lambda var: int(var.Cal.split('_')[1]), axis=1)
                # print(df_1)
            df_1 = df_1[['year', 'week', 'volume']]
            df_1.loc[(df_1['volume'] <= 0), 'volume'] = 0.00001

            # def
            seas_week = {}
            for x in range(1, 53):
                seas_week[x] = 0

            for f_y in df_1.year.unique():
                # print(f_y)
                for m in range(1, 53):
                    # print(m)
                    if len(df_1[(df_1['year'] == f_y) & (df_1['week'] == m)]['volume']) == 0:
                        seas_week[m] = seas_week[m]
                    else:
                        seas_week[m] = seas_week[m] + \
                                       df_1[(df_1['year'] == f_y) & (df_1['week'] == m)]['volume'].values[0] / \
                                       df_1[df_1.year == f_y]['volume'].mean()

            for x in seas_week.keys():
                # print(x, len(df_1[df_1.week == x]))
                seas_week[x] = seas_week[x] / len(df_1[df_1.week == x])
            # print(seas_week)

            df_1['Seas'] = df_1['week'].map(seas_week)

            # print(df_1)
            if no_sales_criteria(df_1.volume, 20):
                print(f"No sales criteria {chain} {l3}")
                continue
            else:
                X = df_1['volume']
                X = X.reset_index(drop=True)

                Y = df_1['volume'].copy()
                quan_res = quantile_range(Y)
                sigma_res = three_sigma_borders(Y)
                Y[(Y.values < quan_res[0])] = quan_res[0]
                Y[(Y.values > quan_res[1])] = quan_res[1]
                t = 1
                X_m = X[:-t]
                train_size = int(len(X_m) * 0.6)
                print(train_size)
                train_X, test_X = X_m[:train_size].to_list(), X_m[train_size:].to_list()

                Y_m = Y[:-t]
                train_Y, test_Y = Y_m[:train_size].to_list(), Y_m[train_size:].to_list()

                Seas = df_1['Seas'].reset_index(drop=True).to_list()[-len(test_X) - t:-t]
                best_params_X = []
                best_params_Y = []

                df_modeling = df_1.assign(set_type='Train')
                df_modeling.iloc[train_size:, 4] = 'Test'
                df_modeling = df_modeling.drop(['Seas'], axis=1)
                df_modeling = df_modeling.assign(cpg=chain)
                df_modeling = df_modeling.assign(l3=l3)
                df_modeling_X = df_modeling.assign(correction='No')
                df_modeling_Y = df_modeling.assign(correction='Yes')

                best_params_founder(
                    SimpleSmooth_Seas(train_X, test_X, Seas, 0.05, df_modeling_X.iloc[:-t], modeling=0),
                    best_params_X)
                best_params_founder(
                    Holt_Seas(train_X, test_X, Seas, 0.1, 0.1, df_modeling_X.iloc[:-t], modeling=0),
                    best_params_X)
                best_params_founder(
                    Holt_Winters(train_X, test_X, 0.2, 0.2, 0.2, df_modeling_X.iloc[:-t], modeling=0,
                                 seasonal_period=52),
                    best_params_X)
                best_params_founder(
                    Arima(train_X, test_X, 15, 2, 2, df_modeling_X.iloc[:-t], modeling=0),
                    best_params_X)

                best_params_founder(
                    SimpleSmooth_Seas(train_Y, test_Y, Seas, 0.05, df_modeling_Y.iloc[:-t], modeling=0),
                    best_params_Y)
                best_params_founder(
                    Holt_Seas(train_Y, test_Y, Seas, 0.1, 0.1, df_modeling_X.iloc[:-t], modeling=0),
                    best_params_Y)
                best_params_founder(
                    Holt_Winters(train_Y, test_Y, 0.2, 0.2, 0.2, df_modeling_X.iloc[:-t], modeling=0,
                                 seasonal_period=52),
                    best_params_Y)
                best_params_founder(
                    Arima(train_Y, test_Y, 15, 2, 2, df_modeling_X.iloc[:-t], modeling=0),
                    best_params_Y)

                print(f"Best models {chain} and {l3}: ", best_params_X)
                print(f"Best models {chain} and {l3} corr: ", best_params_Y)

                # print(df_1)

                period = 75
                df_final = df_1.copy()
                df_final = df_final.drop('Seas', axis=1)
                print(df_final)
                res_X = best_models_fit(X_m, best_params_X, period)
                res_Y = best_models_fit(Y_m, best_params_Y, period)

                df_final = df_final.assign(predict_smoothing=res_X[0][0:len(df_final)])
                df_final = df_final.assign(predict_holt=res_X[1][0:len(df_final)])
                df_final = df_final.assign(predict_arima=res_X[2][0:len(df_final)])
                df_final = df_final.assign(predict_holt_wint=res_X[3][0:len(df_final)])
                df_final = df_final.assign(l3_id=l3)
                df_final = df_final.assign(buyer_id=chain)
                df_final = df_final.assign(region_id='')
                df_final = df_final.assign(sellin_corr=list(Y.values))
                df_final = df_final.assign(predict_smoothing_corr=res_Y[0][0:len(df_final)])
                df_final = df_final.assign(predict_holt_corr=res_Y[1][0:len(df_final)])
                df_final = df_final.assign(predict_arima_corr=res_Y[2][0:len(df_final)])
                df_final = df_final.assign(predict_holt_wint_corr=res_Y[3][0:len(df_final)])
                df_final = df_final.assign(date_upload='2023-12-15 00:00:00.000')
                df_final = df_final.assign(status_id=0)

                len_st = len(df_final)

                cur_Year = df_final.year.max()
                cur_week = df_final.week.tail(1).values[0]
                week = cur_week
                year = cur_Year
                for i in range(period):
                    week = week + 1
                    if week > 52:
                        week = 1
                        year += 1
                    new_row = pd.Series({"year": year,
                                         "week": week,
                                         "volume": 0,
                                         "predict_smoothing": res_X[0][len_st + i],
                                         "predict_holt": res_X[1][len_st + i],
                                         "predict_arima": res_X[2][len_st + i],
                                         "predict_holt_wint": res_X[3][len_st + i],
                                         "l3_id": l3,
                                         "buyer_id": chain,
                                         "region_id": '',
                                         "sellin_corr": 0,
                                         "predict_smoothing_corr": res_Y[0][len_st + i],
                                         "predict_holt_corr": res_Y[1][len_st + i],
                                         "predict_arima_corr": res_Y[2][len_st + i],
                                         "predict_holt_wint_corr": res_Y[3][len_st + i],
                                         "date_upload": '2023-12-15 00:00:00.000',
                                         "status_id": 0})
                    df_final = df_final.append(new_row, ignore_index=True)
                print(df_final)

                df_final['Seas'] = df_final['week'].map(seas_week)

                df_final['predict_smoothing_seas'] = df_final['Seas'] * df_final['predict_smoothing']
                df_final['predict_holt_seas'] = df_final['Seas'] * df_final['predict_holt']

                df_final['predict_smoothing_corr_seas'] = df_final['Seas'] * df_final['predict_smoothing_corr']
                df_final['predict_holt_corr_seas'] = df_final['Seas'] * df_final['predict_holt_corr']

                quan_res = quantile_range(df_1['volume'])

                for column in ['predict_smoothing', 'predict_holt', 'predict_arima',
                               'predict_smoothing_corr', 'predict_holt_corr', 'predict_arima_corr',
                               'predict_smoothing_seas', 'predict_holt_seas', 'predict_smoothing_corr_seas',
                               'predict_holt_corr_seas', 'predict_holt_wint', 'predict_holt_wint_corr']:
                    df_final[column][(df_final[column].values < quan_res[0])] = quan_res[0]
                    df_final[column][(df_final[column].values < 0)] = 0
                    df_final[column][(df_final[column].values > 1000000000000)] = 1000000000000
                # print(df_final)

                df_final = df_final.fillna(0)
                # df_final = df_final.merge(year_calendar, left_on='year', right_on='year', how='left')
                df_final = df_final.rename(columns={'year': 'year_id'})
                df_final = df_final.drop(
                    ['predict_smoothing', 'predict_holt', 'predict_smoothing_corr', 'predict_holt_corr'], axis=1)
                df_final = df_final[
                    ['buyer_id', 'l3_id', 'year_id', 'week', 'volume', 'predict_smoothing_seas',
                     'predict_holt_seas', 'predict_arima', 'predict_holt_wint', 'sellin_corr',
                     'predict_smoothing_corr_seas',
                     'predict_holt_corr_seas', 'predict_arima_corr', 'predict_holt_wint_corr', 'date_upload',
                     'status_id']]

                # print(df_final)
                model_dict = {'Smoothing': 'predict_smoothing_seas',
                              'Holt': 'predict_holt_seas',
                              'ARIMA': 'predict_arima',
                              'Holt-Winters': 'predict_holt_wint',
                              'Smoothing_corr': 'predict_smoothing_corr_seas',
                              'Holt_corr': 'predict_holt_corr_seas',
                              'ARIMA_corr': 'predict_arima_corr',
                              'Holt-Winters_corr': 'predict_holt_wint_corr',
                              'Unknown': 'Unknown'
                              }
                best_model_df = model_dict[define_best_model_test_np(best_params_X, best_params_Y)]
                print("Best model test", best_model_df)
                df_final_st = df_final[:len_st]
                score = 0.
                best_model = 'Unknown'
                for name in ['predict_smoothing_seas', 'predict_holt_seas', 'predict_arima', 'predict_holt_wint',
                             'predict_smoothing_corr_seas', 'predict_holt_corr_seas', 'predict_arima_corr',
                             'predict_holt_wint_corr']:
                    try:
                        if score == 0.:
                            best_model = name
                            score = mean_squared_error(df_final_st.volume, df_final_st[name])
                        if score > mean_squared_error(df_final_st.volume, df_final_st[name]):
                            best_model = name
                            score = mean_squared_error(df_final_st.volume, df_final_st[name])
                    except ValueError:
                        continue
                print("Best model total", best_model)
                df_final['best_model_total'] = best_model
                df_final['best_model'] = best_model_df

                if (best_model_df == 'Unknown') or ((df_final[best_model_df] == 0).all()):
                    df_final['best_model_value'] = df_final[best_model]
                else:
                    df_final['best_model_value'] = df_final[best_model_df]

                print(df_final)

                df_final.to_csv('week_test_res.csv', mode='a')


def metadata():
    raw_data = pd.read_csv('data_raw_1.csv', sep=';', decimal=',', engine='python')
    raw_data['status_id'] = raw_data['status_id'].replace({'Regular': 2, 'Promo': 1})
    raw_data = raw_data.rename(columns={'week': 'week_id'})
    print(raw_data.head())
    # Counting percent for day_of_week
    # days_percent = raw_data[(raw_data.is_archive == 'Active') & (raw_data.volume > 0)].copy()
    days_percent = raw_data[(raw_data.volume > 0)].copy()
    days_percent['Дата: День, Месяц, Год'] = pd.to_datetime(days_percent['Дата: День, Месяц, Год'], format='%d.%m.%Y')
    days_percent['day_of_week'] = days_percent['Дата: День, Месяц, Год'].dt.weekday
    week_day_volume = days_percent.groupby(['day_of_week'])['volume'].sum().reset_index(name='volume_dw')
    week_day_volume['Percent_dw'] = week_day_volume['volume_dw'] / week_day_volume['volume_dw'].sum()

    # Counting percent for down-leveling
    # sku_df = raw_data[(raw_data.year == 2024) & (raw_data.is_archive == 'Active') & (raw_data.volume > 0)].copy()
    sku_df = raw_data[(raw_data.year == 2024) & (raw_data.volume > 0)].copy()
    sku_df = sku_df.groupby(['cpg_id', 'ppg_id', 'Клиент', 'Материал', 'Завод'])['volume'].sum().reset_index(
        name='volume_sku')
    sku_df_total = sku_df.groupby(['cpg_id', 'ppg_id'])['volume_sku'].sum().reset_index(name='volume_sku_total')
    sku_df = pd.merge(sku_df, sku_df_total, left_on=['cpg_id', 'ppg_id'], right_on=['cpg_id', 'ppg_id'], how='left')
    sku_df['Percent_sku'] = sku_df.volume_sku / sku_df.volume_sku_total

    return (raw_data,
            week_day_volume[['day_of_week', 'Percent_dw']],
            sku_df[['cpg_id', 'ppg_id', 'Клиент', 'Материал', 'Завод', 'Percent_sku']])


def data_preparation(raw_data, cpg, ppg, group_type):
    # raw_data_filtered = raw_data[
    #     ((raw_data.cpg_id == cpg) & (raw_data.ppg_id == ppg) &
    #      (raw_data.is_archive == 'Active') & (raw_data.volume > 0))]
    raw_data_filtered = raw_data[
        ((raw_data.cpg_id == cpg) & (raw_data.ppg_id == ppg) &
         (raw_data.volume > 0))]
    if len(raw_data_filtered) > 0:
        if group_type == 'month':
            df = raw_data_filtered[['year', 'month_id', 'status_id', 'volume']]
            df = df.groupby(by=['year', 'month_id', 'status_id'], as_index=False).sum()
            df = df.rename(columns={'month_id': 'month'})
            df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            df = df.drop(['year', 'month'], axis=1)
            df.set_index('date', inplace=True)
            df['volume'] = df['volume'].fillna(0)

        elif group_type == 'week':
            df = raw_data_filtered[['year', 'week_id', 'status_id', 'volume']]
            df = df.groupby(by=['year', 'week_id', 'status_id'], as_index=False).sum()
            df = df.astype({'year': np.int64, 'week_id': np.int64, 'volume': np.float64})
            df['date'] = df.apply(lambda var: get_date_from_year_week(var.year, var.week_id), axis=1)
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop(['year', 'week_id'], axis=1)
            df.set_index('date', inplace=True)
            df['volume'] = df['volume'].fillna(0)
        else:
            df = pd.DataFrame()

    else:
        df = pd.DataFrame()

    return df


def data_visualisation(df, type_date, type_graph):
    if 'plot_rolling' in type_graph:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=df[df.segment == 'raw']['target'], x=df.index, name='volume'))
        fig.add_trace(go.Scatter(y=df[df.segment == 'rolling']['target'], x=df.index, name='rolling_mean'))

        fig.update_layout(title="Данные с скользящее среднее",
                          height=1400,
                          showlegend=True)
    if 'promo_regular' in type_graph:
        df_grouped = df.groupby(['date', 'status_id'])['volume'].sum().unstack(fill_value=0)
        if type_date == 'week':
            df_resampled = df_grouped.resample('W-MON').sum().fillna(0)
        elif type_date == 'month':
            df_resampled = df_grouped.resample('M').sum().fillna(0)
        if 2 not in df_resampled.columns:
            df_resampled[2] = 0
        if 1 not in df_resampled.columns:
            df_resampled[1] = 0
        df_promo_percent = df_resampled.copy()
        df_promo_percent['year'] = df_promo_percent.index.year
        total_promo_sum = df_promo_percent[1].sum()
        total_regular_sum = df_promo_percent[2].sum()
        total_promo_mean = df_promo_percent[1].mean()
        total_regular_mean = df_promo_percent[2].mean()
        total_promo_per = (total_promo_sum / (total_promo_sum + total_regular_sum) * 100).round(2)
        df_promo_percent = df_promo_percent.groupby(by=['year']).agg({1: ['sum', 'mean'],
                                                                      2: ['sum', 'mean']})

        df_promo_percent['promo_%'] = (df_promo_percent[(1, 'sum')] /
                                       (df_promo_percent[(2, 'sum')] + df_promo_percent[(1, 'sum')]) * 100).round(2)
        total_row = pd.DataFrame({
            (1, 'sum'): [total_promo_sum],
            (1, 'mean'): [total_promo_mean],
            (2, 'sum'): [total_regular_sum],
            (2, 'mean'): [total_regular_mean],
            ('promo_%', ''): [total_promo_per]
        }, index=['Total'])
        df_promo_percent = pd.concat([df_promo_percent, total_row])
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.7, 0.3],
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=['', ''],
            specs=[[{"type": "bar"}, ],
                   [{"type": "table"}]]
        )

        fig.add_trace(go.Bar(y=df_resampled[2].values, x=df_resampled.index, name='regular',
                             marker=dict(color='salmon')), row=1, col=1)
        fig.add_trace(go.Bar(y=df_resampled[1].values, x=df_resampled.index, name='promo',
                             marker=dict(color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(y=df_resampled.sum(axis=1).values, x=df_resampled.index, name='total', mode='lines',
                                 line=dict(color='black', width=4)), row=1, col=1)
        fig.add_trace(go.Table(
            header=dict(values=['Year', 'Sum_Promo', 'Mean_Promo', 'Sum_Regular', 'Mean_Regular', 'Promo_%']),
            cells=dict(values=[df_promo_percent.index,
                               df_promo_percent[(1, 'sum')].round(2), df_promo_percent[(1, 'mean')].round(2),
                               df_promo_percent[(2, 'sum')].round(2), df_promo_percent[(2, 'mean')].round(2),
                               df_promo_percent[('promo_%', '')]])
        ), row=2, col=1)

        fig.update_layout(barmode='stack',
                          title="Данные с информацией о доле промо",
                          height=1400,
                          showlegend=True)

    fig.show();


def status_identifactor(data, status_id):
    if status_id == 3:
        df = data.groupby(data.index)['volume'].sum().to_frame()
        df['status_id'] = 3
    else:
        df = data[data.status_id == status_id]
    return df


def TS_transformation(data, type_date, correction, status_id, final_date, rolling_week=4, rolling_month=2):
    if type_date == 'month':
        dd_st = status_identifactor(data, status_id=status_id)
        rf_df = dd_st.copy()
        quan_res = quantile_range(dd_st['volume'])
        print("Quantile range:", quan_res)
        dd_st = dd_st.drop(['status_id'], axis=1)
        dd_st = dd_st.resample('MS').sum().fillna(0)
        date_range = pd.date_range(start=dd_st.index[0], end=final_date, freq='MS')
        additional_data = pd.DataFrame(0, index=date_range, columns=dd_st.columns)
        dd_st = dd_st[(dd_st.index >= additional_data.index.min()) & (dd_st.index <= additional_data.index.max())]
        dd_st = dd_st.combine_first(additional_data)
        dd_st.index.name = 'date'
        if correction == 1:
            print("Rows for correction\n", dd_st[dd_st['volume'] > quan_res[1]])
            dd_st[dd_st['volume'] > quan_res[1]] = quan_res[1]
        dd_st['timestamp'] = dd_st.index
        dd_st['segment'] = 'raw'
        data_row = dd_st[['timestamp', 'volume', 'segment']].reset_index().drop('date', axis=1)
        data_diff = data_row.copy(deep=True)
        data_diff['rolling'] = data_diff[['volume']].rolling(rolling_month, min_periods=1).mean()
        data_diff['volume'] = data_diff['rolling']
        data_diff = data_diff.drop('rolling', axis=1)
        data_diff['segment'] = 'rolling'
        df = pd.concat([data_row, data_diff], axis=0)
        df['volume'] = df['volume'].fillna(0)
        df['volume'] = df['volume'].replace(0, 0.000001)
        df = df.rename(columns={'volume': 'target'})
        # data_visualisation(df, 'month', ['plot_rolling'])
        df = TSDataset.to_dataset(df)
        ts_1 = TSDataset(df, freq='MS')
        df_pandas = ts_1.to_pandas()
        df_filled = df_pandas.fillna(0)
        ts_filled = TSDataset(df_filled, freq='MS')

    if type_date == 'week':
        dd_st = status_identifactor(data, status_id=status_id)
        rf_df = dd_st.copy()
        quan_res = quantile_range(dd_st['volume'])
        print("Quantile range:", quan_res)
        dd_st = dd_st.drop(['status_id'], axis=1)
        dd_st = dd_st.resample('W-MON').sum().fillna(0)
        date_range = pd.date_range(start=dd_st.index[0], end=final_date, freq='W-MON')
        additional_data = pd.DataFrame(0, index=date_range, columns=dd_st.columns)
        dd_st = dd_st[(dd_st.index >= additional_data.index.min()) & (dd_st.index <= additional_data.index.max())]
        dd_st = dd_st.combine_first(additional_data)
        dd_st.index.name = 'date'
        if correction == 1:
            print("Rows for correction\n", dd_st[dd_st['volume'] > quan_res[1]])
            dd_st[dd_st['volume'] > quan_res[1]] = quan_res[1]
        dd_st['timestamp'] = dd_st.index
        dd_st['segment'] = 'raw'
        data_row = dd_st[['timestamp', 'volume', 'segment']].reset_index().drop('date', axis=1)
        data_diff = data_row.copy(deep=True)
        data_diff['rolling'] = data_diff[['volume']].rolling(rolling_week, min_periods=1).mean()  # .shift()
        data_diff['volume'] = data_diff['rolling']
        data_diff = data_diff.drop('rolling', axis=1)
        data_diff['segment'] = 'rolling'
        df = pd.concat([data_row, data_diff], axis=0)
        df['volume'] = df['volume'].fillna(0)
        df['volume'] = df['volume'].replace(0, 0.000001)
        df = df.rename(columns={'volume': 'target'})
        # data_visualisation(df, ['week'], ['plot_rolling'])
        df = TSDataset.to_dataset(df)
        ts_1 = TSDataset(df, freq='W-MON')
        df_pandas = ts_1.to_pandas()
        df_filled = df_pandas.fillna(0)
        ts_filled = TSDataset(df_filled, freq='W-MON')
    return ts_filled, rf_df[['volume']]


def skip_option(df, column_name, zeros):
    if df.loc[:, column_name][-zeros:].sum() == 0.000001 * zeros:
        return True
    else:
        return False


def seasonality_calculation(data, tag, type_date):
    if type_date == 'week':
        df = data.df.copy()
        df['week'] = df.index.isocalendar().week
        df['year'] = df.index.year
        df = df[df.year == 2023]
        if df[(tag, 'target')].mean() == 0:
            df = pd.DataFrame()
        else:
            mean_week = df[(tag, 'target')].mean()
            df['mean_week'] = mean_week
            df['seasonal_coefficient'] = df[(tag, 'target')] / mean_week
        df = df[['week', 'seasonal_coefficient']]
        df.columns = [''.join(col) for col in df.columns]
    if type_date == 'month':
        df = data.df.copy()
        df['month'] = df.index.month
        df['year'] = df.index.year
        df = df[df.year == 2023]
        if df[(tag, 'target')].mean() == 0:
            df = pd.DataFrame()
        else:
            mean_month = df[(tag, 'target')].mean()
            df['mean_month'] = mean_month
            df['seasonal_coefficient'] = df[(tag, 'target')] / mean_month
        df = df[['month', 'seasonal_coefficient']]
        df.columns = [''.join(col) for col in df.columns]
    return df


def prophet_modeling(train, test, type_date, changepoint, seasonality, horizon, score):
    print("Start Prophet")
    for ch_p in np.arange(0.1, changepoint, 0.5):
        for s_p in np.arange(10., seasonality, 5):
            try:
                train1 = TSDataset(df=train.df.copy(), freq=train.freq)
                if type_date == 'week':
                    prophet_model = ProphetModel(daily_seasonality=False,
                                                 weekly_seasonality=True,
                                                 yearly_seasonality=False,
                                                 changepoint_prior_scale=ch_p,
                                                 seasonality_prior_scale=s_p,
                                                 uncertainty_samples=100,
                                                 # growth='logistic',
                                                 additional_seasonality_params=[
                                                     {'name': 'month', 'period': 12, 'fourier_order': 12}]
                                                 )
                if type_date == 'month':
                    prophet_model = ProphetModel(daily_seasonality=False,
                                                 weekly_seasonality=False,
                                                 yearly_seasonality=True,
                                                 changepoint_prior_scale=ch_p,
                                                 seasonality_prior_scale=s_p,
                                                 uncertainty_samples=100,
                                                 # growth='logistic',
                                                 additional_seasonality_params=[
                                                     {'name': 'month', 'period': 12, 'fourier_order': 6}]
                                                 )
                prophet_model.fit(train1)
                future_ts = train1.make_future(horizon)
                forecast_ts = prophet_model.forecast(future_ts)
                forecast_df = forecast_ts.to_pandas(False)
                # forecast_df.loc[forecast_df[('raw', 'target')] < 0, ('raw', 'target')] = 0
                # forecast_df.loc[forecast_df[('rolling', 'target')] < 0, ('rolling', 'target')] = 0
                score_raw = mean_squared_error(forecast_df[('raw', 'target')], test.loc[:, ('raw', 'target')]) ** 0.5
                score_rolling = mean_squared_error(forecast_df[('rolling', 'target')],
                                                   test.loc[:, ('raw', 'target')]) ** 0.5
                # print(ch_p, s_p)
                # print('rmse_raw', score_raw)
                # print('rmse_rolling', score_rolling)
                score = score.append({'model': 'prophet',
                                      'data': 'raw',
                                      'par': str(ch_p) + "_" + str(s_p) + '_raw',
                                      'score': score_raw},
                                     ignore_index=True)
                score = score.append({'model': 'prophet',
                                      'data': 'rolling',
                                      'par': str(ch_p) + "_" + str(s_p) + '_rolling',
                                      'score': score_rolling},
                                     ignore_index=True)
            except Exception as e:
                print(e)
                continue
    return score


def arima_modeling(train, test, tag, p_max, q_max, d_max, horizon, score):
    print('Start Arima')
    for d in range(0, d_max + 1):
        for q in range(0, q_max + 1):
            for p in range(1, p_max + 1, 1):
                try:
                    arima_model = ARIMA(train.df[(tag, 'target')], order=(p, q, d)).fit()
                    fcast = arima_model.forecast(horizon)
                    # print(statistics.pstdev(fcast) / statistics.mean(fcast))
                    # print(statistics.pstdev(test.df[(tag, 'target')]) / statistics.mean(test.df[(tag, 'target')]))
                    if (statistics.pstdev(fcast) / statistics.mean(fcast) >=
                            statistics.pstdev(test.df[('raw', 'target')]) / statistics.mean(
                                test.df[('raw', 'target')]) * (
                                    1 - 0.5)):
                        score_value = (mean_squared_error(fcast, test.df[('raw', 'target')])) ** 0.5
                        score = score.append({'model': 'arima',
                                              'data': tag,
                                              'par': str(p) + "_" + str(q) + "_" + str(d) + f'_{tag}',
                                              'score': score_value},
                                             ignore_index=True)
                except Exception:
                    continue
    return score


def holt_winters_modeling(train, test, type_date, tag, step1, step2, step3, horizon, score):
    print('Start Holt-Winters')
    for alpha in range(0, 100, int(step1 * 100)):
        for beta in range(0, 100, int(step2 * 100)):
            for gamma in range(0, 100, int(step3 * 100)):
                for tr in ['add', 'mul', None]:
                    for damp in [True, False]:
                        for ss in ['add', 'mul']:
                            try:
                                if type_date == 'month':
                                    model = ExponentialSmoothing(train.df[(tag, 'target')],
                                                                 seasonal_periods=12,
                                                                 trend=tr,
                                                                 damped_trend=damp,
                                                                 seasonal=ss,
                                                                 initialization_method='estimated'
                                                                 ).fit(smoothing_level=alpha / 100,
                                                                       smoothing_trend=beta / 100,
                                                                       smoothing_seasonal=gamma / 100)
                                if type_date == 'week':
                                    model = ExponentialSmoothing(train.df[(tag, 'target')],
                                                                 seasonal_periods=52,
                                                                 trend=tr,
                                                                 damped_trend=damp,
                                                                 seasonal=ss,
                                                                 initialization_method='estimated'
                                                                 ).fit(smoothing_level=alpha / 100,
                                                                       smoothing_trend=beta / 100,
                                                                       smoothing_seasonal=gamma / 100)
                                fcast = model.forecast(horizon)
                                if (statistics.pstdev(fcast) / statistics.mean(fcast) >=
                                        statistics.pstdev(test.df[('raw', 'target')]) / statistics.mean(
                                            test.df[('raw', 'target')]) * (
                                                1 - 0.5)):
                                    score_value = (mean_squared_error(fcast, test.df[('raw', 'target')])) ** 0.5
                                    score = score.append({'model': 'holt_winters',
                                                          'data': tag,
                                                          'par': str(alpha) + "_" + str(beta) + "_" + str(gamma) + "_"
                                                                 + str(tr) + "_" + str(damp) + "_" + str(
                                                              ss) + f'_{tag}',
                                                          'score': score_value},
                                                         ignore_index=True)

                            except Exception:
                                continue
    return score


def smoothing_modeling(train, test, type_date, seas, tag, step, horizon, score):
    print('Start Simple Smoothing')
    for alpha in range(0, 100, int(step * 100)):
        try:
            model = SimpleExpSmoothing(train.df[(tag, 'target')], initialization_method="heuristic").fit(
                smoothing_level=alpha / 100, optimized=False)
            fcast = model.forecast(horizon)
            fcast_df = pd.DataFrame(fcast, columns=['forecast'])
            if type_date == 'week':
                fcast_df['week'] = fcast_df.index.isocalendar().week
                fcast_df['original_index'] = fcast_df.index
                fcast_df = pd.merge(fcast_df, seas, left_on='week', right_on='week', how='left')
                fcast_df.set_index('original_index', inplace=True)
                fcast_df = fcast_df.fillna({'seasonal_coefficient': 0})
            if type_date == 'month':
                fcast_df['month'] = fcast_df.index.month
                fcast_df['original_index'] = fcast_df.index
                fcast_df = pd.merge(fcast_df, seas, left_on='month', right_on='month', how='left')
                fcast_df.set_index('original_index', inplace=True)
                fcast_df = fcast_df.fillna({'seasonal_coefficient': 0})
            fcast_df['forecast_seas'] = fcast_df['forecast'] * fcast_df['seasonal_coefficient']
            # fcast_df.loc[fcast_df['forecast_seas'] < 0, 'forecast_seas'] = 0
            # print(statistics.pstdev(fcast_df['forecast_seas']) / statistics.mean(fcast_df['forecast_seas']))
            # print(statistics.pstdev(test.df[(tag, 'target')]) / statistics.mean(test.df[(tag, 'target')]))
            if (statistics.pstdev(fcast_df['forecast_seas']) / statistics.mean(fcast_df['forecast_seas']) >=
                    statistics.pstdev(test.df[(tag, 'target')]) / statistics.mean(test.df[(tag, 'target')]) * (
                            1 - 0.5)):
                score_value = (mean_squared_error(fcast_df['forecast_seas'], test.df[('raw', 'target')])) ** 0.5
                score = score.append({'model': 'smoothing',
                                      'data': tag,
                                      'par': str(alpha) + f'_{tag}',
                                      'score': score_value},
                                     ignore_index=True)
        except Exception:
            continue
    return score


def holt_modeling(train, test, type_date, seas, tag, step1, step2, horizon, score):
    print('Start Holt')
    for alpha in range(0, 100, int(step1 * 100)):
        for beta in range(0, 100, int(step2 * 100)):
            for exp in [True, False]:
                for damp in [False]:
                    try:
                        model = Holt(train.df[(tag, 'target')],
                                     exponential=exp,
                                     damped_trend=damp,
                                     initialization_method="estimated").fit(smoothing_level=alpha / 100,
                                                                            smoothing_trend=beta / 100)
                        fcast = model.forecast(horizon)
                        fcast_df = pd.DataFrame(fcast, columns=['forecast'])
                        if type_date == 'week':
                            fcast_df['week'] = fcast_df.index.isocalendar().week
                            fcast_df['original_index'] = fcast_df.index
                            fcast_df = pd.merge(fcast_df, seas, left_on='week', right_on='week', how='left')
                            fcast_df.set_index('original_index', inplace=True)
                            fcast_df = fcast_df.fillna({'seasonal_coefficient': 0})
                        if type_date == 'month':
                            fcast_df['month'] = fcast_df.index.month
                            fcast_df['original_index'] = fcast_df.index
                            fcast_df = pd.merge(fcast_df, seas, left_on='month', right_on='month', how='left')
                            fcast_df.set_index('original_index', inplace=True)
                            fcast_df = fcast_df.fillna({'seasonal_coefficient': 0})
                        fcast_df['forecast_seas'] = fcast_df['forecast'] * fcast_df['seasonal_coefficient']
                        # fcast_df.loc[fcast_df['forecast_seas'] < 0, 'forecast_seas'] = 0
                        # print(statistics.pstdev(fcast_df['forecast_seas']) / statistics.mean(fcast_df['forecast_seas']))
                        # print(statistics.pstdev(test.df[(tag, 'target')]) / statistics.mean(test.df[(tag, 'target')]))
                        if (statistics.pstdev(fcast_df['forecast_seas']) / statistics.mean(fcast_df['forecast_seas']) >=
                                statistics.pstdev(test.df[('raw', 'target')]) / statistics.mean(
                                    test.df[('raw', 'target')]) * (
                                        1 - 0.5)):
                            score_value = (mean_squared_error(fcast_df['forecast_seas'],
                                                              test.df[('raw', 'target')])) ** 0.5
                            score = score.append({'model': 'holt',
                                                  'data': tag,
                                                  'par': str(alpha) + '_' + str(beta) + '_' +
                                                         str(exp) + '_' + str(damp) + f'_{tag}',
                                                  'score': score_value},
                                                 ignore_index=True)
                    except Exception:
                        continue
    return score


def best_models(data, score, seas, type_date, models_list, horizon):
    final_df = data.df
    final_df = final_df.drop([('rolling', 'target')], axis=1)
    max_value = final_df[('raw', 'target')].max() * 5
    final_df.columns = ['_'.join(col) for col in final_df.columns]
    best_model_df = pd.DataFrame(columns=['model', 'score', 'score_total'])
    score_t = 0
    for mdl in models_list:
        score_model = score[score['model'] == mdl]
        for tag in ['raw', 'rolling']:
            score_model_tag = score_model[score_model.data == tag]
            if len(score_model_tag) == 0:
                final_df[mdl + '_' + tag] = 0
            else:
                best_params = score_model_tag[score_model_tag.score == score_model_tag.score.min()].par.values[0]
                if mdl == 'prophet':
                    try:
                        ch_p_f, s_p_f, type_data = best_params.split('_')
                        data1 = TSDataset(df=data.df.copy(), freq=data.freq)
                        if type_date == 'week':
                            prophet_model_final = ProphetModel(daily_seasonality=False,
                                                               weekly_seasonality=True,
                                                               yearly_seasonality=False,
                                                               changepoint_prior_scale=ch_p_f,
                                                               seasonality_prior_scale=s_p_f,
                                                               uncertainty_samples=100,
                                                               # growth='logistic',
                                                               additional_seasonality_params=[
                                                                   {'name': 'month',
                                                                    'period': 12,
                                                                    'fourier_order': 12}],
                                                               )
                        if type_date == 'month':
                            prophet_model_final = ProphetModel(daily_seasonality=False,
                                                               weekly_seasonality=False,
                                                               yearly_seasonality=False,
                                                               changepoint_prior_scale=ch_p_f,
                                                               seasonality_prior_scale=s_p_f,
                                                               uncertainty_samples=100,
                                                               # growth='logistic',
                                                               additional_seasonality_params=[
                                                                   {'name': 'month', 'period': 12, 'fourier_order': 6}]
                                                               )
                        prophet_model_final.fit(data1)
                        fcst = prophet_model_final.predict(data1)
                        score_t = mean_squared_error(fcst.loc[:, (type_data, 'target')],
                                                     data1.loc[:, (type_data, 'target')]) ** 0.5
                        future_ts = data1.make_future(horizon)
                        forecast_ts = prophet_model_final.forecast(future_ts)
                        forecast_df = forecast_ts.to_pandas(False)
                        forecast_df = pd.DataFrame(forecast_df[(type_data, 'target')])
                        forecast_df.columns = [mdl + '_' + tag]
                        forecast_df.loc[forecast_df[mdl + '_' + tag] < 0, mdl + '_' + tag] = 0
                        forecast_df.loc[forecast_df[mdl + '_' + tag] > max_value, mdl + '_' + tag] = max_value
                        final_df = pd.merge(final_df, forecast_df, left_index=True, right_index=True, how='outer')
                        final_df = final_df.fillna(0)
                    except Exception:
                        final_df[mdl + '_' + tag] = 0
                if mdl == 'arima':
                    try:
                        p, q, d, tag = best_params.split('_')
                        arima_model = ARIMA(data.df[(tag, 'target')], order=(int(p), int(q), int(d))).fit()
                        fcst = arima_model.fittedvalues
                        score_t = mean_squared_error(fcst,
                                                     data.loc[:, (tag, 'target')]) ** 0.5
                        fcast = arima_model.forecast(horizon)
                        forecast_df = pd.DataFrame(fcast)
                        forecast_df.columns = [mdl + '_' + tag]
                        forecast_df.loc[forecast_df[mdl + '_' + tag] < 0, mdl + '_' + tag] = 0
                        final_df = pd.merge(final_df, forecast_df, left_index=True, right_index=True, how='outer')
                        final_df = final_df.fillna(0)
                    except Exception as e:
                        print(e)
                        final_df[mdl + '_' + tag] = 0
                if mdl == 'smoothing':
                    try:
                        alpha, tag = best_params.split('_')
                        smoothing_model = SimpleExpSmoothing(data.df[(tag, 'target')],
                                                             initialization_method="heuristic").fit(
                            smoothing_level=int(alpha) / 100, optimized=False)
                        fcast = smoothing_model.forecast(horizon)
                        fcast_df = pd.DataFrame(fcast, columns=['forecast'])
                        if type_date == 'week':
                            fcast_df['week'] = fcast_df.index.isocalendar().week
                            fcast_df['original_index'] = fcast_df.index
                            fcast_df = pd.merge(fcast_df, seas, left_on='week', right_on='week', how='left')
                            fcast_df.set_index('original_index', inplace=True)
                            fcast_df = fcast_df.fillna({'seasonal_coefficient': 0})
                        if type_date == 'month':
                            fcast_df['month'] = fcast_df.index.month
                            fcast_df['original_index'] = fcast_df.index
                            fcast_df = pd.merge(fcast_df, seas, left_on='month', right_on='month', how='left')
                            fcast_df.set_index('original_index', inplace=True)
                            fcast_df = fcast_df.fillna({'seasonal_coefficient': 0})
                        fcast_df['forecast_seas'] = fcast_df['forecast'] * fcast_df['seasonal_coefficient']
                        fcast_df.loc[fcast_df['forecast_seas'] < 0, 'forecast_seas'] = 0
                        fcast_df.loc[fcast_df['forecast_seas'] > max_value, 'forecast_seas'] = max_value
                        forecast_df = pd.DataFrame(fcast_df['forecast_seas'])
                        forecast_df.columns = [mdl + '_' + tag]
                        forecast_df.loc[forecast_df[mdl + '_' + tag] < 0, mdl + '_' + tag] = 0
                        final_df = pd.merge(final_df, forecast_df, left_index=True, right_index=True, how='outer')
                        final_df = final_df.fillna(0)
                    except Exception:
                        final_df[mdl + '_' + tag] = 0
                if mdl == 'holt':
                    try:
                        alpha, beta, exp, damp, tag = best_params.split('_')
                        holt_model = Holt(data.df[(tag, 'target')],
                                          exponential=exp,
                                          damped_trend=damp,
                                          initialization_method="estimated").fit(smoothing_level=float(alpha) / 100,
                                                                                 smoothing_trend=float(beta) / 100)
                        fcast = holt_model.forecast(horizon)
                        fcast_df = pd.DataFrame(fcast, columns=['forecast'])
                        if type_date == 'week':
                            fcast_df['week'] = fcast_df.index.isocalendar().week
                            fcast_df['original_index'] = fcast_df.index
                            fcast_df = pd.merge(fcast_df, seas, left_on='week', right_on='week', how='left')
                            fcast_df.set_index('original_index', inplace=True)
                            fcast_df = fcast_df.fillna({'seasonal_coefficient': 0})
                        if type_date == 'month':
                            fcast_df['month'] = fcast_df.index.month
                            fcast_df['original_index'] = fcast_df.index
                            fcast_df = pd.merge(fcast_df, seas, left_on='month', right_on='month', how='left')
                            fcast_df.set_index('original_index', inplace=True)
                            fcast_df = fcast_df.fillna({'seasonal_coefficient': 0})
                        fcast_df['forecast_seas'] = fcast_df['forecast'] * fcast_df['seasonal_coefficient']
                        fcast_df.loc[fcast_df['forecast_seas'] < 0, 'forecast_seas'] = 0
                        fcast_df.loc[fcast_df['forecast_seas'] > max_value, 'forecast_seas'] = max_value
                        forecast_df = pd.DataFrame(fcast_df['forecast_seas'])
                        forecast_df.columns = [mdl + '_' + tag]
                        forecast_df.loc[forecast_df[mdl + '_' + tag] < 0, mdl + '_' + tag] = 0
                        final_df = pd.merge(final_df, forecast_df, left_index=True, right_index=True, how='outer')
                        final_df = final_df.fillna(0)
                    except Exception:
                        final_df[mdl + '_' + tag] = 0
                if mdl == 'holt_winters':
                    try:
                        alpha, beta, gamma, tr, damp, ss, tag = best_params.split('_')
                        # print(alpha, beta, gamma, tr, damp, ss, tag)
                        if type_date == 'week':
                            hw_model = ExponentialSmoothing(data.df[(tag, 'target')],
                                                            seasonal_periods=52,
                                                            trend=tr,
                                                            damped_trend=damp,
                                                            seasonal=ss,
                                                            initialization_method='estimated'
                                                            ).fit(smoothing_level=float(alpha) / 100,
                                                                  smoothing_trend=float(beta) / 100,
                                                                  smoothing_seasonal=float(gamma) / 100)
                        if type_date == 'month':
                            hw_model = ExponentialSmoothing(data.df[(tag, 'target')],
                                                            seasonal_periods=12,
                                                            trend=tr,
                                                            damped_trend=damp,
                                                            seasonal=ss,
                                                            initialization_method='estimated'
                                                            ).fit(smoothing_level=float(alpha) / 100,
                                                                  smoothing_trend=float(beta) / 100,
                                                                  smoothing_seasonal=float(gamma) / 100)
                        fcast = hw_model.forecast(horizon)
                        forecast_df = pd.DataFrame(fcast)
                        forecast_df.columns = [mdl + '_' + tag]
                        forecast_df.loc[forecast_df[mdl + '_' + tag] < 0, mdl + '_' + tag] = 0
                        forecast_df.loc[forecast_df[mdl + '_' + tag] > max_value, mdl + '_' + tag] = max_value
                        final_df = pd.merge(final_df, forecast_df, left_index=True, right_index=True, how='outer')
                        final_df = final_df.fillna(0)
                    except Exception:
                        final_df[mdl + '_' + tag] = 0

                best_model_df = best_model_df.append({'model': mdl + '_' + tag,
                                                      'score': score_model_tag.score.min(),
                                                      'score_total': score_t},
                                                     ignore_index=True)
    return final_df, best_model_df


def best_model_growth_update(data, best_params, growth, fall, width):
    len_st = len(data[data.raw_target > 0])
    last_year_index = max(len_st - width, 0)

    last_year_volume = data[last_year_index:len_st].raw_target.sum()
    # print(last_year_volume)
    best_name = 'simplest'

    len_of_last_year = len(data[last_year_index:len_st])
    best_mdl_list = best_params.model.values.tolist()
    for bm in best_mdl_list:
        next_year_volume = data[len_st:len_st + len_of_last_year][bm].sum()
        print(f"Growth with model {bm} is {next_year_volume / last_year_volume}")
        if ((next_year_volume / last_year_volume > (1 + growth) or next_year_volume / last_year_volume < (1 - fall)) or
                next_year_volume == 0):
            continue
        else:
            best_name = bm
            break
    return best_name


# date_type = 'week'
# flag = 1
# number_of_zeros = {'week': 24, 'month': 6}
# final_fact_date = pd.to_datetime('2024-09-29')
# final_forecast_date = pd.to_datetime('2025-04-01')
# simplest_model_range = {'week': 26, 'month': 6}
# growing_range = {'week': 26, 'month': 6}
# horizon_frcst = {'week': 14, 'month': 6}

# raw_dataset, day_of_week_df, sku_percent = metadata()
# print(day_of_week_df)


def main_prediction_v2(date_type, cpg_list, ppg_list, status_id, time_connection,
                       final_fact_date, rolling_dict, number_of_zeros, horizon_frcst,
                       simplest_model_range, growing_range):
    raw_dataset, year_list, day_of_week_df = metadata_DB(cpg_list, ppg_list, status_name=status_id)
    final_fact_date = pd.to_datetime(final_fact_date)
    time_connection = datetime.strptime(time_connection, '%Y-%m-%d')
    if len(cpg_list) == 0:
        cpg_wlist = raw_dataset.cpg_id.unique()
    else:
        cpg_wlist = cpg_list
    for cpg_d in cpg_wlist:
        if len(ppg_list) == 0:
            ppg_wlist = raw_dataset[raw_dataset.cpg_id == cpg_d].ppg_id.unique()
        else:
            ppg_wlist = ppg_list
        # print(ppg_wlist)
        for ppg_d in ppg_wlist:
            print(f'Start for {cpg_d} and {ppg_d}')
            dd = data_preparation(raw_dataset, cpg_d, ppg_d, group_type=date_type)
            # print(dd)
            if len(dd) > 0:
                # data_visualisation(dd, type_date=date_type, type_graph=['promo_regular'])
                mdls_list = ['prophet', 'arima', 'smoothing', 'holt', 'holt_winters']
                for st in status_id:
                    print("Status ", st)
                    if (((len(dd[dd.status_id == st]) > 0) and (
                            dd[dd.status_id == st].index.min() < final_fact_date)) or
                            ((st == 3) and (dd.index.min() < final_fact_date))):
                        ts, raw_fact = TS_transformation(dd,
                                                         type_date=date_type,
                                                         correction=1,
                                                         status_id=st,
                                                         final_date=final_fact_date,
                                                         rolling_month=rolling_dict['month'],
                                                         rolling_week=rolling_dict['week'])
                        # print(ts)
                        skip = skip_option(ts, column_name=('raw', 'target'), zeros=number_of_zeros[date_type])
                        seas_df = seasonality_calculation(ts, tag='rolling', type_date=date_type)
                        score_df = pd.DataFrame(columns=['model', 'data', 'par', 'score'])
                        if skip or len(ts.df) <= 1:
                            print(f'Skipped {cpg_d} and {ppg_d} with status {st}')
                            total_table, best_params_df = best_models(data=ts, score=score_df, seas=seas_df,
                                                                      type_date=date_type, models_list=mdls_list,
                                                                      horizon=horizon_frcst[date_type])
                        else:
                            HORIZON = min(max(int(len(ts.df) * 0.25), 1), int(len(ts.df)) - 1)
                            print(HORIZON)
                            train_ts, test_ts = ts.train_test_split(test_size=HORIZON)
                            score_df = holt_winters_modeling(train=train_ts, test=test_ts, type_date=date_type,
                                                             tag='raw', step1=0.25, step2=0.25, step3=0.25,
                                                             horizon=HORIZON, score=score_df)
                            score_df = holt_winters_modeling(train=train_ts, test=test_ts, type_date=date_type,
                                                             tag='rolling', step1=0.25, step2=0.25, step3=0.25,
                                                             horizon=HORIZON, score=score_df)
                            score_df = holt_modeling(train=train_ts, test=test_ts, type_date=date_type, seas=seas_df,
                                                     tag='raw', step1=0.1, step2=0.1, horizon=HORIZON,
                                                     score=score_df)
                            score_df = holt_modeling(train=train_ts, test=test_ts, type_date=date_type, seas=seas_df,
                                                     tag='rolling', step1=0.1, step2=0.1, horizon=HORIZON,
                                                     score=score_df)
                            score_df = smoothing_modeling(train=train_ts, test=test_ts, type_date=date_type,
                                                          seas=seas_df,
                                                          tag='raw', step=0.05, horizon=HORIZON,
                                                          score=score_df)
                            score_df = smoothing_modeling(train=train_ts, test=test_ts, type_date=date_type,
                                                          seas=seas_df,
                                                          tag='rolling', step=0.05, horizon=HORIZON,
                                                          score=score_df)
                            score_df = arima_modeling(train_ts, test_ts, 'raw', 9, 2, 2,
                                                      horizon=HORIZON, score=score_df)
                            score_df = arima_modeling(train_ts, test_ts, 'rolling', 9, 2, 2,
                                                      horizon=HORIZON, score=score_df)
                            score_df = prophet_modeling(train=train_ts, test=test_ts, type_date=date_type,
                                                        changepoint=1.2, seasonality=16., horizon=HORIZON,
                                                        score=score_df)
                            total_table, best_params_df = best_models(data=ts, score=score_df, seas=seas_df,
                                                                      type_date=date_type, models_list=mdls_list,
                                                                      horizon=horizon_frcst[date_type])

                        simplest_index = max(len(ts.df) - simplest_model_range[date_type], 0)
                        total_table['simplest'] = total_table[simplest_index:len(ts.df)].raw_target.mean()
                        best_params_df['holt_flag'] = best_params_df['model'].apply(
                            lambda x: 1 if x in ['holt_raw', 'holt_rolling'] else 0)
                        best_params_df = best_params_df.sort_values(by=['holt_flag', 'score']).drop('holt_flag', axis=1)
                        best_model_name = best_model_growth_update(total_table, best_params_df, 0.8, 0.4,
                                                                   growing_range[date_type])
                        total_table['best_model_name'] = '' if skip else best_model_name
                        total_table['best_model_value'] = 0 if skip else total_table[best_model_name]

                        total_table['status_id'] = st # 'Regular' if st == 2 else 'Promo' if st == 1 else 'Total'
                        total_table['cpg'] = cpg_d
                        total_table['ppg'] = ppg_d
                        total_table['correction'] = 'Yes'  # if flag == 1 else 'No'
                        total_table['date_upload'] = time_connection
                        total_table = pd.merge(total_table, raw_fact, left_index=True, right_index=True, how='left')
                        total_table['volume'] = total_table['volume'].fillna(0)
                        total_table.index.name = 'index'
                        print(total_table)
                        filename = time_connection.strftime("%d%m%y")
                        file_tag = 'result'
                        filename += "___" + str(file_tag) + ".csv"
                        filename = "data/" + filename
                        print(filename)
                        # total_table.reset_index().to_csv(filename, decimal=',', index=False, mode='a')
                        # total_table_test = total_table.tail(horizon_frcst[date_type])
                        # metric_df = pd.DataFrame()
                        # metric_df.at[0, 'cpg'] = cpg_d
                        # metric_df.at[0, 'ppg'] = ppg_d
                        # metric_df.at[0, 'status_id'] = 'Regular' if st == 2 else 'Promo' if st == 1 else 'Total'
                        # for cols in ['prophet_raw', 'prophet_rolling', 'arima_raw', 'arima_rolling', 'smoothing_raw',
                        #              'smoothing_rolling', 'holt_raw', 'holt_rolling', 'holt_winters_raw',
                        #              'holt_winters_rolling', 'simplest', 'best_model_value']:
                        #     metric_df.at[0, cols] = 1 - (
                        #                 (total_table_test[cols] - total_table_test['volume']).abs().sum() /
                        #                 total_table_test['volume'].sum())
                        # print(metric_df)

if __name__ == '__main__':
    main_prediction_v2(date_type='week',
                       cpg_list=[1425],
                       ppg_list=[322],
                       status_id=[2, 3],
                       time_connection='2025-03-04',
                       final_fact_date='2024-11-30',
                       rolling_dict={'week': 4, 'month': 2},
                       number_of_zeros={'week': 24, 'month': 6},
                       horizon_frcst={'week': 52, 'month': 6},
                       simplest_model_range={'week': 26, 'month': 6},
                       growing_range={'week': 26, 'month': 6})

# print(sku_percent.head(10))
# cpg_list = ['HoReCa', 'АО "ДИКСИ ЮГ"', 'АО "ТАНДЕР"', 'АО ТД ПЕРЕКРЕСТОК', 'Дискаунтеры', 'Дистрибьюторы',
#             'Локальные сети (Прочее)', 'Локальные сети (ТОП)', 'ООО "О`КЕЙ"', 'ООО "ОНЛАЙН-ГИПЕРМАРКЕТ"',
#             'ООО "УМНЫЙ РИТЕЙЛ"', 'Розница', 'Собственные сети', 'СОЮЗ СВ. ИОАННА ВОИНА ООО', 'ООО "АШАН"',
#             'ООО "ИНТЕРНЕТ РЕШЕНИЯ"', 'ООО "ЯНДЕКС.ЛАВКА"', 'ООО "ГИПЕРГЛОБУС"', 'ООО "АГРОТОРГ"',
#             'ООО Лента', 'ООО "МЕТРО КЭШ ЭНД КЕРРИ"', 'АТАК ООО', 'ООО "АГРОАСПЕКТ"', 'КОПЕЙКА-МОСКВА ООО',
#             'ООО "СЛАДКАЯ ЖИЗНЬ Н.Н."', 'ГОРОДСКОЙ СУПЕРМАРКЕТ ООО']
#
# cpg_list = [7228, 7274]
#cpg_list = ['Собственные сети']
#'СЕРВЕЛАТ ГОСТ ЗП п / а 750г В / У ОХЛ'

# ppg_list = ['КОЛБАСА ПОЛУКОПЧЕНАЯ Чесночная ф/о охл В/У 375г', 'КОЛБАСА СЕРВЕЛАТ Коньячный охл в/к в/у 375',
#             'КОЛБАСА Сервелат Финский в/к в/у 375', 'Колбаса_Докторская_вареная_охл',
#             'Колбаса_Докторская_вареная_охл_470г', 'Колбаса_Классическая_охл_п/а_~1200г',
#             'Колбаса_Классическая_охл_п/а_470г', 'Колбаса_Краковская_полукопченая_охл_н/о_430г',
#             'Колбаса_Молочная_вареная_охл', 'Колбаса_с_молоком_вареная_охл',
#             'Колбаса_Сервелат_варено-копч._охл_фиброуз_375г', 'Колбаса_Филейная_вареная_охл_п/а',
#             'РЕБРЫШКИ по-домашнему 500г охл в/у', 'СЕРВЕЛАТ ГОСТ вк ф/о 300гр В/У ОХЛ',
#             'СЕРВЕЛАТ ГОСТ ЗП п/а 750г В/У ОХЛ', 'Сервелат Коньячный в/к нарезка ОХЛ ГЗМС 100гр',
#             'СЕРВЕЛАТ МРАМОРНЫЙ ВУ ОХЛ 330г', 'Сервелат Финский в/к нарезка ОХЛ ГЗМС 100гр',
#             'СОСИСКИ МОЛОЧНЫЕ ц/о 350гр ГЗМС ОХЛ', 'СОСИСКИ ОРИГИНАЛЬНЫЕ п/а 350г ОХЛ ГЗМС', 'Сосиски_Венские_охл_ц/о',
#             'Сосиски_Молочные_охл_ц/о_400г', 'Сосиски_Сочные_охл_п/а']

#ppg_list = ['Сосиски_Сочные_охл_п/а']
# for cpg_d in cpg_list:
#     ppg_list = raw_dataset[raw_dataset.cpg_id == cpg_d].ppg_id.unique()
#     print(ppg_list)
#     #ppg_list = [167, 152, 119]
#     for ppg_d in ppg_list:
#         print(f'Start for {cpg_d} and {ppg_d}')
#         dd = data_preparation(raw_dataset, cpg_d, ppg_d, group_type=date_type)
#         if len(dd) > 0:
#             #data_visualisation(dd, type_date=date_type, type_graph=['promo_regular'])
#             mdls_list = ['prophet', 'arima', 'smoothing', 'holt', 'holt_winters']
#             for st in [2, 3]:
#                 print("Status ", st)
#                 if (((len(dd[dd.status_id == st]) > 0) and (dd[dd.status_id == st].index.min() < final_fact_date)) or
#                         ((st == 3) and (dd.index.min() < final_fact_date))):
#                     ts, raw_fact = TS_transformation(dd,
#                                                      type_date=date_type,
#                                                      correction=flag,
#                                                      status_id=st,
#                                                      final_date=final_fact_date)
#                     skip = skip_option(ts, column_name=('raw', 'target'), zeros=number_of_zeros[date_type])
#                     seas_df = seasonality_calculation(ts, tag='rolling', type_date=date_type)
#                     #print(seas_df)
#                     score_df = pd.DataFrame(columns=['model', 'data', 'par', 'score'])
#                     if skip or len(ts.df) <= 1:
#                         print(f'Skipped {cpg_d} and {ppg_d} with status {st}')
#                         total_table, best_params_df = best_models(data=ts, score=score_df, seas=seas_df,
#                                                                   type_date=date_type, models_list=mdls_list,
#                                                                   horizon=horizon_frcst[date_type])
#                     else:
#                         HORIZON = min(max(int(len(ts.df) * 0.25), 1), int(len(ts.df)) - 1)
#                         print(HORIZON)
#                         train_ts, test_ts = ts.train_test_split(test_size=HORIZON)
#                         score_df = holt_winters_modeling(train=train_ts, test=test_ts, type_date=date_type,
#                                                          tag='raw', step1=0.25, step2=0.25, step3=0.25,
#                                                          horizon=HORIZON, score=score_df)
#                         score_df = holt_winters_modeling(train=train_ts, test=test_ts, type_date=date_type,
#                                                          tag='rolling', step1=0.25, step2=0.25, step3=0.25,
#                                                          horizon=HORIZON, score=score_df)
#                         score_df = holt_modeling(train=train_ts, test=test_ts, type_date=date_type, seas=seas_df,
#                                                  tag='raw', step1=0.1, step2=0.1, horizon=HORIZON,
#                                                  score=score_df)
#                         score_df = holt_modeling(train=train_ts, test=test_ts, type_date=date_type, seas=seas_df,
#                                                  tag='rolling', step1=0.1, step2=0.1, horizon=HORIZON,
#                                                  score=score_df)
#                         score_df = smoothing_modeling(train=train_ts, test=test_ts, type_date=date_type, seas=seas_df,
#                                                       tag='raw', step=0.05, horizon=HORIZON,
#                                                       score=score_df)
#                         score_df = smoothing_modeling(train=train_ts, test=test_ts, type_date=date_type, seas=seas_df,
#                                                       tag='rolling', step=0.05, horizon=HORIZON,
#                                                       score=score_df)
#                         score_df = arima_modeling(train_ts, test_ts, 'raw', 9, 2, 2,
#                                                   horizon=HORIZON, score=score_df)
#                         score_df = arima_modeling(train_ts, test_ts, 'rolling', 9, 2, 2,
#                                                   horizon=HORIZON, score=score_df)
#                         score_df = prophet_modeling(train=train_ts, test=test_ts, type_date=date_type,
#                                                     changepoint=1.2, seasonality=16., horizon=HORIZON, score=score_df)
#                         total_table, best_params_df = best_models(data=ts, score=score_df, seas=seas_df,
#                                                                   type_date=date_type, models_list=mdls_list,
#                                                                   horizon=horizon_frcst[date_type])
#                     print(best_params_df)
#
#                     simplest_index = max(len(ts.df) - simplest_model_range[date_type], 0)
#                     total_table['simplest'] = total_table[simplest_index:len(ts.df)].raw_target.mean()
#                     best_params_df['holt_flag'] = best_params_df['model'].apply(
#                         lambda x: 1 if x in ['holt_raw', 'holt_rolling'] else 0)
#                     best_params_df = best_params_df.sort_values(by=['holt_flag', 'score']).drop('holt_flag', axis=1)
#                     best_model_name = best_model_growth_update(total_table, best_params_df, 0.8, 0.4,
#                                                                growing_range[date_type])
#                     total_table['best_model_name'] = '' if skip else best_model_name
#                     total_table['best_model_value'] = 0 if skip else total_table[best_model_name]
#
#                     total_table['status_id'] = 'Regular' if st == 2 else 'Promo' if st == 1 else 'Total'
#                     total_table['cpg'] = cpg_d
#                     total_table['ppg'] = ppg_d
#                     total_table['correction'] = 'Yes' if flag == 1 else 'No'
#                     total_table = pd.merge(total_table, raw_fact, left_index=True, right_index=True, how='left')
#                     total_table['volume'] = total_table['volume'].fillna(0)
#                     total_table.index.name = 'index'
#                     #print(total_table)
#                     total_table_test = total_table.tail(horizon_frcst[date_type])
#                     metric_df = pd.DataFrame()
#                     metric_df.at[0, 'cpg'] = cpg_d
#                     metric_df.at[0, 'ppg'] = ppg_d
#                     metric_df.at[0, 'status_id'] = 'Regular' if st == 2 else 'Promo' if st == 1 else 'Total'
#                     for cols in ['prophet_raw', 'prophet_rolling', 'arima_raw', 'arima_rolling', 'smoothing_raw',
#                                  'smoothing_rolling', 'holt_raw', 'holt_rolling', 'holt_winters_raw',
#                                  'holt_winters_rolling', 'simplest', 'best_model_value']:
#                         metric_df.at[0, cols] = 1 - ((total_table_test[cols] - total_table_test['volume']).abs().sum() /
#                                                      total_table_test['volume'].sum())
#                     print(metric_df)
#                     total_table.reset_index().to_csv('president_weekly.csv', decimal=',', index=False, mode='a')
#                     metric_df.to_csv('president_metrcis.csv', decimal=',', index=False, mode='a')
#                     # start_date = '2024-07-01'
#                     # end_date = '2024-09-02'
#                     # total_table_daily = total_table.loc[start_date:end_date]
#                     # total_table_daily = total_table_daily.resample('D').ffill().reset_index()
#                     # total_table_daily['index'] = pd.to_datetime(total_table_daily['index'], format='%d.%m.%Y')
#                     # total_table_daily['day_of_week'] = total_table_daily['index'].dt.weekday
#                     # total_table_daily = pd.merge(total_table_daily,
#                     #                              day_of_week_df,
#                     #                              left_on='day_of_week',
#                     #                              right_on='day_of_week',
#                     #                              how='left')
#                     # total_table_daily = pd.merge(total_table_daily,
#                     #                              sku_percent,
#                     #                              left_on=['cpg', 'ppg'],
#                     #                              right_on=['cpg_id', 'ppg_id'],
#                     #                              how='left')
#                     # total_table_daily = total_table_daily.rename(columns={'index': 'date'})
#                     # for cols in ['prophet_raw', 'prophet_rolling', 'arima_raw', 'arima_rolling', 'smoothing_raw',
#                     #              'smoothing_rolling', 'holt_raw', 'holt_rolling', 'simplest', 'best_model_value']:
#                     #     total_table_daily[cols] = (total_table_daily[cols] * total_table_daily['Percent_dw'] *
#                     #                                total_table_daily['Percent_sku'])
#                     # total_table_daily = total_table_daily.drop(['raw_target', 'volume', 'cpg_id', 'ppg_id','day_of_week',
#                     #                                             'Percent_dw', 'Percent_sku'], axis=1)
#                     # total_table_daily.to_csv('miratorg_test_3_250125.csv', decimal=',', index=False, mode='a')
