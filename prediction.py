import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import statistics
import pyodbc
from sqlalchemy import create_engine
import urllib
import math
from datetime import datetime


def current_version():
    return "Forecast_3_models.ver_1.3"


# Func for defining quantile range
def quantile_range(data):
    q25 = data.quantile(q=0.25)
    q75 = data.quantile(q=0.75)
    delta = q75 - q25
    boundaries = (q25 - 1. * delta, q75 + 1.5 * delta)
    return boundaries


# Func for defining 3 sigma range
def three_sigma_borders(data):
    low = data.mean() - 3 * data.std()
    high = data.mean() + 3 * data.std()
    res = (low, high)
    return res


# Research seasonal coefficients
def seasonal(df, n):
    full_year = []
    seas_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}
    for y in df.year.unique():
        if len(df[df.year == y]) == n:
            full_year.append(y)
    for f_y in full_year:
        for m in range(1, 13):
            seas_dict[m] = seas_dict[m] + df[(df['year'] == f_y) & (df['month_id'] == m)]['volume'].values[0] / \
                           df[df.year == f_y]['volume'].mean()
    if len(full_year) > 0:
        for x in seas_dict.keys():
            seas_dict[x] = seas_dict[x] / len(full_year)
    return seas_dict


def set_value(row_number, assigned_value):
    return assigned_value[row_number]


# Form full list month and year between start point and finish point
def sample_calendar(start_year, start_month, final_year, final_month):
    res = []
    s = 1
    f = 12
    for y in range(start_year, final_year + 1):
        if y == start_year:
            s = start_month
        elif y == final_year:
            f = final_month
        for x in range(s, f + 1):
            res.append([str(y) + '_' + str(x)])
            s = 1
            f = 12
    df = pd.DataFrame(res)
    return df


# Func for counting quality metrics
def metrics(df, buyer, group, target_column, prediction_columns, periods):
    res = pd.DataFrame(columns=['chain'] + ['group'] + ['model'] + ['metric'] + periods)
    for predict in prediction_columns:
        mape = []
        baes = []
        for p in periods:
            m = round((abs(df[predict][-15:-15 + p] - df[target_column][-15:-15 + p]).sum()) / df[target_column][
                                                                                               -15:-15 + p].sum(), 2)
            b = round(df[predict][-15:-15 + p].sum() / df[target_column][-15:-15 + p].sum() - 1, 2)
            baes.append(b)
            mape.append(m)
        res.loc[len(res)] = [buyer, group, predict, 'BAES'] + baes
        res.loc[len(res)] = [buyer, group, predict, 'mape'] + mape
    return res


# Simple Smoothing forecast
def SimpleSmooth_Seas(train, test, seas, step):
    print('Start SimpleSmooth')
    for i in seas:
        if math.isnan(i) or i == 0:
            print("Seasonal is not defined")
            return {}
    error_dict_X_S = {}
    for alpha in range(0, 100, int(step * 100)):
        try:
            model = SimpleExpSmoothing(train, initialization_method="heuristic").fit(
                smoothing_level=alpha / 100, optimized=False)
            fcast = model.forecast(len(test))
            fcast_seas = []
            for x, y in zip(fcast, seas):
                fcast_seas.append(x * y)
            if (statistics.pstdev(fcast_seas) >= statistics.pstdev(test) * (1 - 0.90)) and \
                    (statistics.pstdev(model.fittedvalues) >= statistics.pstdev(train) * (1 - 0.90)):
                error_dict_X_S['Smoothing_' + str(alpha / 100)] = [
                    mean_absolute_error(train, model.fittedvalues),
                    mean_absolute_error(test, fcast_seas)]
        except ValueError:
            continue
    model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
    fcast = model.forecast(len(test))
    fcast_seas = []
    for x, y in zip(fcast, seas):
        fcast_seas.append(x * y)
    if (statistics.pstdev(fcast_seas) >= statistics.pstdev(test) * (1 - 0.90)) and \
            (statistics.pstdev(model.fittedvalues) >= statistics.pstdev(train) * (1 - 0.90)):
        error_dict_X_S['Smoothing_' + str(model.model.params["smoothing_level"])] = [
            mean_absolute_error(train, model.fittedvalues),
            mean_absolute_error(test, fcast_seas)]
    return error_dict_X_S


# Holt forecast
def Holt_Seas(train, test, seas, step1, step2):
    print('Start Holt')
    for i in seas:
        if math.isnan(i) or i == 0:
            print("Seasonal is not defined")
            return {}
    error_dict_X_S = {}
    for alpha in range(0, 100, int(step1 * 100)):
        for beta in range(0, 100, int(step2 * 100)):
            for exp in [True, False]:
                for damp in [True, False]:
                    try:
                        model = Holt(train, exponential=exp, damped_trend=damp, initialization_method="estimated").fit(
                            smoothing_level=alpha / 100, smoothing_trend=beta / 100)
                        fcast = model.forecast(len(test))
                        fcast_seas = []
                        for x, y in zip(fcast, seas):
                            fcast_seas.append(x * y)
                        if (statistics.pstdev(fcast_seas) >= statistics.pstdev(test) * (1 - 0.50)) \
                                and (statistics.pstdev(model.fittedvalues) >= statistics.pstdev(train) * (1 - 0.50)):
                            error_dict_X_S['Holt_' + str(exp) + '/' + str(damp) + '/' + str(alpha / 100) + '/' + str(
                                beta / 100)] = [
                                mean_absolute_error(train, model.fittedvalues),
                                mean_absolute_error(test, fcast_seas)]
                    except OverflowError:
                        continue
    return error_dict_X_S


# Arima forecast
def Arima(train, test, p_max, q_max, d_max):
    print('Start Arima')
    error_dict_X_S = {}
    for d in range(1, d_max + 1):
        for q in range(1, q_max + 1):
            for p in range(1, p_max + 1, 1):
                try:
                    model = ARIMA(train, order=(p, q, d)).fit()
                    fcast = model.forecast(len(test))
                    if (statistics.pstdev(fcast) >= statistics.pstdev(test) * (1 - 0.50)) \
                            and (statistics.pstdev(model.fittedvalues) >= statistics.pstdev(train) * (1 - 0.50)):
                        error_dict_X_S['ARIMA_' + str(p) + '/' + str(q) + '/' + str(d)] = [
                            mean_absolute_error(train, model.fittedvalues),
                            mean_absolute_error(test, fcast)]
                except np.linalg.LinAlgError:
                    continue
                except IndexError:
                    continue
    return error_dict_X_S


# Finding of best_params from error dictionaries
def best_params_founder(err_dict, best_params):
    if len(err_dict) > 0:
        init = sorted([x[0] for x in err_dict.values()])
        threshold = 0.4
        best_train_err = init[0] * (1 + threshold)
        init_1 = sorted([x[1] for x in err_dict.values() if x[0] <= best_train_err])
        for keys, values in err_dict.items():
            if (values[1] == init_1[0]) and (keys not in best_params):
                add_str = keys + "_" + str(init_1[0])
                best_params.append(add_str)
    return best_params


def best_models_fit(df, best_params, period):
    res = []
    if len(best_params) == 1:
        print('This one')
        # res.append([0] * (len(df) + period + 1))
        res.append([0 for x in range(len(df) + period + 1)])
        res.append([0 for x in range(len(df) + period + 1)])
    if len(best_params) == 2:
        print('This one')
        res.append([0 for x in range(len(df) + period + 1)])
    if len(best_params) == 0:
        print('This one')
        res.append([0 for x in range(len(df) + period + 1)])
        res.append([0 for x in range(len(df) + period + 1)])
        res.append([0 for x in range(len(df) + period + 1)])
    for elem in best_params:
        model_name = elem.split("_")[0]
        model_params = elem.split("_")[1]
        if model_name == 'Smoothing':
            model_smoothing = SimpleExpSmoothing(df, initialization_method="heuristic").fit(
                smoothing_level=float(model_params), optimized=False)
            fcast_smoothing = model_smoothing.forecast(period)
            res.append([0] + list(model_smoothing.fittedvalues) + list(fcast_smoothing))

        elif model_name == 'Holt':
            model_holt = Holt(df, exponential=model_params.split('/')[0],
                              damped_trend=model_params.split('/')[1],
                              initialization_method="estimated").fit(
                smoothing_level=float(model_params.split('/')[2]),
                smoothing_trend=float(model_params.split('/')[3]))
            fcast_holt = model_holt.forecast(period)
            res.append([0] + list(model_holt.fittedvalues) + list(fcast_holt))
        elif model_name == 'ARIMA':
            try:
                model_arima = ARIMA(df, order=(
                    int(model_params.split('/')[0]),
                    int(model_params.split('/')[1]),
                    int(model_params.split('/')[2]))).fit()
                fcast_arima = model_arima.forecast(period + 1)
                res.append(list(model_arima.fittedvalues) + list(fcast_arima))
            except np.linalg.LinAlgError:
                print("Something is not good")
                res.append([0 for x in range(len(df) + period + 1)])
            except IndexError:
                print("Something is not good")
                res.append([0 for x in range(len(df) + period + 1)])
    # print(res)
    return res


def define_best_model_test(best_params_X, best_params_Y):
    best_score = -1
    best_model = "Unknown"
    for elem in best_params_X:
        model_name = elem.split("_")[0]
        model_score = float(elem.split("_")[2])
        if best_score < 0 or best_score > model_score:
            best_score = model_score
            best_model = model_name
    for elem in best_params_Y:
        model_name = elem.split("_")[0]
        model_score = float(elem.split("_")[2])
        if best_score < 0 or best_score > model_score:
            best_score = model_score
            best_model = model_name + "_corr"
    return best_model


def main_prediction(chain_list, category_list, channel, time_connection, status_name=0):
    if status_name == 0:
        status_name_act = 3
    else:
        status_name_act = status_name

    # time_connection = datetime.now()
    print('Start', time_connection)

    connect_str = ("Driver={SQL Server Native Client 11.0};"
                   "Server=109.73.41.150;"
                   "Database=PromoPlannerMVPPresident;"
                   "UID=Web_User;"
                   "PWD=sPEP12bk0;")
    connection = pyodbc.connect(connect_str)
    cursor = connection.cursor()

    quoted = urllib.parse.quote_plus(connect_str)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={quoted}')

    print('Time of connection', datetime.now() - time_connection)

    chain_fulllist = pd.read_sql("SELECT [id] FROM [PromoPlannerMVPPresident].[dbo].[_spr_address_chain]",
                                 connection)

    category_fulllist = pd.read_sql("SELECT [id] FROM [PromoPlannerMVPPresident].[dbo].[_spr_sku_l3]",
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

    data = pd.read_sql(f"SELECT * FROM [dbo].[SalesInWeek] "
                       f"WHERE chain_id in {chain_str} and l3_id in {category_str}"
                       f"and channel1_id = {channel}", connection)
    print(f"Was extracted {len(data)} string")
    if status_name > 0:
        data = data.loc[(data['status_id'] == status_name)]
    print(data)

    year_calendar = pd.read_sql("SELECT [id], [year] FROM [PromoPlannerMVPPresident].[dbo].[_spr_date_year]",
                                connection)

    year_calendar = year_calendar.astype({'id': np.int64, 'year': np.int64})

    print('Time of download', datetime.now() - time_connection)

    df_new_1 = data.loc[(data['chain_name'].notna()) | (data['l3_name'].notna())]
    df_new_1 = df_new_1[['year', 'year_id', 'month_id', 'volume', 'chain_id', 'l3_id', 'chain_name', 'l3_name']]
    chain_name = df_new_1.chain_id.value_counts().index.to_list()
    print(chain_name)
    print(df_new_1)

    for chain in chain_name:
        df_final_full = pd.DataFrame(columns=['l3_id', 'chain_id', 'year_id', 'month_id', 'volume',
                                              'predict_smoothing_seas', 'predict_holt_seas',
                                              'predict_arima', 'sellin_corr', 'predict_smoothing_corr_seas',
                                              'predict_holt_corr_seas', 'predict_arima_corr', 'date_upload',
                                              'best_model_test', 'best_model', 'status_id'])
        l3_name = df_new_1[df_new_1.chain_id == chain].l3_id.value_counts().index.to_list()
        # l3_name = [19.0, 18.0, 17.0, 21.0, 20.0, 15.0, 16.0, 14 .0, 13.0]
        # print(l3_name)
        for l3 in l3_name:
            print(f'Start prediction {chain} for group {l3}')

            df_new_1_2 = df_new_1[(df_new_1.chain_id == chain) & (df_new_1.l3_id == l3)]
            df_new_1_2 = df_new_1_2[['year', 'month_id', 'volume']]
            df_new_1_2 = df_new_1_2.groupby(by=['year', 'month_id'], as_index=False).sum()
            df_new_1_2 = df_new_1_2.astype({'year': np.int64, 'month_id': np.int64, 'volume': np.float64})

            if len(df_new_1_2) < 3:
                print("Not enough length of dataset - ", len(df_new_1_2))
                continue

            df_new_1_2['Cal'] = df_new_1_2.apply(lambda var: str(int(var.year)) + '_' + str(int(var.month_id)), axis=1)
            cals = sample_calendar(df_new_1_2.year.min(),
                                   df_new_1_2[df_new_1_2.year == df_new_1_2.year.min()].month_id.min(),
                                   df_new_1_2.year.max(),
                                   df_new_1_2[df_new_1_2.year == df_new_1_2.year.max()].month_id.max())
            df_new_1_2 = df_new_1_2.merge(cals, left_on='Cal', right_on=0, how='right')
            df_new_1_2['year'] = df_new_1_2.apply(lambda var: int(var.Cal.split('_')[0]), axis=1)
            df_new_1_2['month_id'] = df_new_1_2.apply(lambda var: int(var.Cal.split('_')[1]), axis=1)
            df_new_1_2['volume'] = df_new_1_2['volume'].fillna(0)
            df_new_1_2 = df_new_1_2.drop(['Cal', 0], axis=1)

            df_new_1_2['Seas'] = df_new_1_2['month_id'].map(seasonal(df_new_1_2, 12))

            df_new_1_2 = df_new_1_2.sort_values(by=['year', 'month_id'], ascending=True)

            df_new_1_2.loc[(df_new_1_2['volume'] <= 0), 'volume'] = 0.000001

            X = df_new_1_2['volume']
            X = X.reset_index(drop=True)

            Y = df_new_1_2['volume'].copy()
            quan_res = quantile_range(Y)
            sigma_res = three_sigma_borders(Y)
            Y[(Y.values < quan_res[0])] = quan_res[0]
            Y[(Y.values > quan_res[1])] = quan_res[1]

            t = 1
            X_m = X[:-t]
            train_size = int(len(X_m) * 0.6)
            train_X, test_X = X_m[:train_size].to_list(), X_m[train_size:].to_list()

            Y_m = Y[:-t]
            train_Y, test_Y = Y_m[:train_size].to_list(), Y_m[train_size:].to_list()

            Seas = df_new_1_2['Seas'].reset_index(drop=True).to_list()[-len(test_X) - t:-t]

            best_params_X = []
            best_params_Y = []

            best_params_founder(SimpleSmooth_Seas(train_X, test_X, Seas, 0.05), best_params_X)
            best_params_founder(Holt_Seas(train_X, test_X, Seas, 0.1, 0.1), best_params_X)
            best_params_founder(Arima(train_X, test_X, 14, 2, 2), best_params_X)

            best_params_founder(SimpleSmooth_Seas(train_Y, test_Y, Seas, 0.05), best_params_Y)
            best_params_founder(Holt_Seas(train_Y, test_Y, Seas, 0.1, 0.1), best_params_Y)
            best_params_founder(Arima(train_Y, test_Y, 14, 2, 2), best_params_Y)

            period = 18
            df_final = df_new_1_2.copy()
            df_final.drop('Seas', axis=1)
            res_X = best_models_fit(X_m, best_params_X, period)
            res_Y = best_models_fit(Y_m, best_params_Y, period)

            df_final = df_final.assign(predict_smoothing=res_X[0][0:len(df_final)])
            df_final = df_final.assign(predict_holt=res_X[1][0:len(df_final)])
            df_final = df_final.assign(predict_arima=res_X[2][0:len(df_final)])
            df_final = df_final.assign(l3_id=l3)
            df_final = df_final.assign(chain_id=chain)
            df_final = df_final.assign(region_id='')
            df_final = df_final.assign(sellin_corr=list(Y.values))
            df_final = df_final.assign(predict_smoothing_corr=res_Y[0][0:len(df_final)])
            df_final = df_final.assign(predict_holt_corr=res_Y[1][0:len(df_final)])
            df_final = df_final.assign(predict_arima_corr=res_Y[2][0:len(df_final)])
            df_final = df_final.assign(date_upload=time_connection)
            df_final = df_final.assign(status_id=status_name_act)

            len_st = len(df_final)

            cur_Year = df_final.year.max()
            cur_Month = df_final.month_id.tail(1).values[0]
            month = cur_Month
            year = cur_Year
            for i in range(period):
                month = month + 1
                if month > 12:
                    month = 1
                    year += 1
                new_row = pd.Series({"year": year,
                                     "month_id": month,
                                     "volume": 0,
                                     "predict_smoothing": res_X[0][len_st + i],
                                     "predict_holt": res_X[1][len_st + i],
                                     "predict_arima": res_X[2][len_st + i],
                                     "l3_id": l3,
                                     "chain_id": chain,
                                     "region_id": '',
                                     "sellin_corr": 0,
                                     "predict_smoothing_corr": res_Y[0][len_st + i],
                                     "predict_holt_corr": res_Y[1][len_st + i],
                                     "predict_arima_corr": res_Y[2][len_st + i],
                                     "date_upload": time_connection,
                                     "status_id": status_name_act})
                df_final = df_final.append(new_row, ignore_index=True)

            df_final['Seas'] = df_final['month_id'].map(seasonal(df_new_1_2, 12))
            df_final['predict_smoothing_seas'] = df_final['Seas'] * df_final['predict_smoothing']
            df_final['predict_holt_seas'] = df_final['Seas'] * df_final['predict_holt']

            df_final['predict_smoothing_corr_seas'] = df_final['Seas'] * df_final['predict_smoothing_corr']
            df_final['predict_holt_corr_seas'] = df_final['Seas'] * df_final['predict_holt_corr']

            quan_res = quantile_range(df_new_1_2['volume'])

            for column in ['predict_smoothing', 'predict_holt', 'predict_arima',
                           'predict_smoothing_corr', 'predict_holt_corr', 'predict_arima_corr',
                           'predict_smoothing_seas', 'predict_holt_seas', 'predict_smoothing_corr_seas',
                           'predict_holt_corr_seas']:
                df_final[column][(df_final[column].values < quan_res[0])] = quan_res[0]
                df_final[column][(df_final[column].values < 0)] = 0

            df_final = df_final.fillna(0)
            df_final = df_final.merge(year_calendar, left_on='year', right_on='year', how='left')
            df_final = df_final.rename(columns={'id': 'year_id'})
            df_final = df_final.drop(
                ['predict_smoothing', 'predict_holt', 'predict_smoothing_corr', 'predict_holt_corr'], axis=1)
            df_final = df_final[
                ['chain_id', 'l3_id', 'year_id', 'month_id', 'volume', 'predict_smoothing_seas',
                 'predict_holt_seas', 'predict_arima', 'sellin_corr', 'predict_smoothing_corr_seas',
                 'predict_holt_corr_seas', 'predict_arima_corr', 'date_upload', 'status_id']]

            # print(df_final)
            model_dict = {'Smoothing': 'predict_smoothing_seas',
                          'Holt': 'predict_holt_seas',
                          'ARIMA': 'predict_arima',
                          'Smoothing_corr': 'predict_smoothing_corr_seas',
                          'Holt_corr': 'predict_holt_corr_seas',
                          'ARIMA_corr': 'predict_arima_corr',
                          'Unknown': 'Unknown'
                          }
            best_model_df = model_dict[define_best_model_test(best_params_X, best_params_Y)]
            print(best_model_df)
            df_final_st = df_final[:len_st]
            score = 0.
            best_model = 'Unknown'
            for name in ['predict_smoothing_seas', 'predict_holt_seas', 'predict_arima',
                         'predict_smoothing_corr_seas', 'predict_holt_corr_seas', 'predict_arima_corr']:
                try:
                    if score == 0.:
                        best_model = name
                        score = mean_squared_error(df_final_st.volume, df_final_st[name])
                    if score > mean_squared_error(df_final_st.volume, df_final_st[name]):
                        best_model = name
                        score = mean_squared_error(df_final_st.volume, df_final_st[name])
                except ValueError:
                    continue

            df_final['best_model_test'] = best_model_df
            df_final['best_model'] = best_model

            df_final_full = df_final_full.append(df_final)
            print(df_final_full)

        df_final_full = df_final_full.astype({'l3_id': np.int64,
                                              'chain_id': np.int64,
                                              'year_id': np.int64,
                                              'month_id': np.int64,
                                              'volume': np.float64,
                                              'predict_smoothing_seas': np.float64,
                                              'predict_holt_seas': np.float64,
                                              'predict_arima': np.float64,
                                              'sellin_corr': np.float64,
                                              'predict_smoothing_corr_seas': np.float64,
                                              'predict_holt_corr_seas': np.float64,
                                              'predict_arima_corr': np.float64,
                                              'date_upload': np.datetime64,
                                              'best_model_test': np.str_,
                                              'best_model': np.str_,
                                              'status_id': np.int64})

        print(df_final_full)

        with engine.connect() as cnn:
            df_final_full.to_sql('forecast_g_raw', schema='dbo', con=cnn, if_exists='append', index=False)
            print(f'Download {chain} was successful')

            print('Time of download full category prediction', datetime.now() - time_connection)
            break
            sql_split_forecast = "EXEC split_forecast_g @DateToLoad=?"
            params = (time_connection)
            cursor.execute(sql_split_forecast, params)
            print('Full time', datetime.now() - time_connection)

    connection.close()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass
    # main_prediction(buyer_list=[], category_list=[], channel=18)
