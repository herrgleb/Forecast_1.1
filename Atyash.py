from prediction import sample_calendar, seasonal, quantile_range, best_params_founder, Arima, \
    Holt_Seas, SimpleSmooth_Seas, best_models_fit, define_best_model_test
import pandas as pd
from datetime import datetime
import pyodbc
import numpy as np
from sklearn.metrics import mean_squared_error


def main_prediction_3(time_connection, status_name=0):
    if status_name == 0:
        status_name_act = 3
    else:
        status_name_act = status_name

    connect_str = ("Driver={SQL Server Native Client 11.0};"
                   "Server=109.73.41.150;"
                   "Database=PromoPlannerMVPPresident;"
                   "UID=Web_User;"
                   "PWD=sPEP12bk0;")
    connection = pyodbc.connect(connect_str)

    data = pd.read_excel('Atyashevo.xlsb', sheet_name='python', engine='pyxlsb', index_col=0)
    # print(data)

    if status_name > 0:
        data = data.loc[(data['status_id'] == status_name)]
    print(data)

    year_calendar = pd.read_sql("SELECT [id], [year] FROM [PromoPlannerMVPPresident].[dbo].[_spr_date_year]",
                                connection)

    year_calendar = year_calendar.astype({'id': np.int64, 'year': np.int64})
    year_calendar = year_calendar.append({'id': 8, 'year': 2018}, ignore_index=True)
    print(year_calendar)

    print('Time of download', datetime.now() - time_connection)

    df_new_1 = data.loc[(data['buyer_name'].notna()) | (data['l3_name'].notna())]
    df_new_1 = df_new_1[['year', 'year_id', 'month_id', 'volume', 'buyer_id', 'l3_id', 'buyer_name', 'l3_name']]
    buyer_name = df_new_1.buyer_id.value_counts().index.to_list()
    # buyer_name = [1]
    print(buyer_name)
    print(df_new_1)

    for buyer in buyer_name:
        l3_name = df_new_1[df_new_1.buyer_id == buyer].l3_id.value_counts().index.to_list()
        # print(l3_name)
        for l3 in l3_name:
            print(f'Start prediction {buyer} for group {l3}')

            df_new_1_2 = df_new_1[(df_new_1.buyer_id == buyer) & (df_new_1.l3_id == l3)]
            df_new_1_2 = df_new_1_2[['year', 'month_id', 'volume']]
            df_new_1_2 = df_new_1_2.groupby(by=['year', 'month_id'], as_index=False).sum()
            df_new_1_2 = df_new_1_2.astype({'year': np.int64, 'month_id': np.int64, 'volume': np.float64})
            print(df_new_1_2)
            quan_res = quantile_range(df_new_1_2['volume'])
            print("quantile_1", quan_res)

            if len(df_new_1_2) < 3:
                print("Not enough length of dataset - ", len(df_new_1_2))
                continue

            df_new_1_2['Cal'] = df_new_1_2.apply(lambda var: str(int(var.year)) + '_' + str(int(var.month_id)), axis=1)
            cals = sample_calendar(df_new_1_2.year.min(),
                                   df_new_1_2[df_new_1_2.year == df_new_1_2.year.min()].month_id.min(),
                                   2023,
                                   5)
            # print(cals)
            df_new_1_2 = df_new_1_2.merge(cals, left_on='Cal', right_on=0, how='right')
            df_new_1_2['year'] = df_new_1_2.apply(lambda var: int(var.Cal.split('_')[0]), axis=1)
            df_new_1_2['month_id'] = df_new_1_2.apply(lambda var: int(var.Cal.split('_')[1]), axis=1)
            df_new_1_2['volume'] = df_new_1_2['volume'].fillna(0)
            df_new_1_2 = df_new_1_2.drop(['Cal', 0], axis=1)

            df_new_1_2['Seas'] = df_new_1_2['month_id'].map(seasonal(df_new_1_2, 12))

            df_new_1_2 = df_new_1_2.sort_values(by=['year', 'month_id'], ascending=True)

            df_new_1_2.loc[(df_new_1_2['volume'] <= 0), 'volume'] = 0.000001

            print(df_new_1_2)

            X = df_new_1_2['volume']
            X = X.reset_index(drop=True)

            Y = df_new_1_2['volume'].copy()
            # quan_res = quantile_range(Y)
            Y[(Y.values < quan_res[0])] = quan_res[0]
            Y[(Y.values > quan_res[1])] = quan_res[1]

            t = 0
            X_m = X
            train_size = int(len(X_m) * 0.6)
            train_X, test_X = X_m[:train_size].to_list(), X_m[train_size:].to_list()

            Y_m = Y
            train_Y, test_Y = Y_m[:train_size].to_list(), Y_m[train_size:].to_list()

            Seas = df_new_1_2['Seas'].reset_index(drop=True).to_list()[-len(test_X) - t:]

            best_params_X = []
            best_params_Y = []
            print(X_m)
            print(Seas)

            best_params_founder(SimpleSmooth_Seas(train_X, test_X, Seas, 0.05), best_params_X)
            best_params_founder(Holt_Seas(train_X, test_X, Seas, 0.1, 0.1), best_params_X)
            best_params_founder(Arima(train_X, test_X, 14, 2, 2), best_params_X)

            best_params_founder(SimpleSmooth_Seas(train_Y, test_Y, Seas, 0.05), best_params_Y)
            best_params_founder(Holt_Seas(train_Y, test_Y, Seas, 0.1, 0.1), best_params_Y)
            best_params_founder(Arima(train_Y, test_Y, 14, 2, 2), best_params_Y)

            period = 7
            df_final = df_new_1_2.copy()
            df_final.drop('Seas', axis=1)
            res_X = best_models_fit(X_m, best_params_X, period)
            res_Y = best_models_fit(Y_m, best_params_Y, period)

            df_final = df_final.assign(predict_smoothing=res_X[0][0:len(df_final)])
            df_final = df_final.assign(predict_holt=res_X[1][0:len(df_final)])
            df_final = df_final.assign(predict_arima=res_X[2][0:len(df_final)])
            df_final = df_final.assign(l3_id=l3)
            df_final = df_final.assign(buyer_id=buyer)
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
                                     "buyer_id": buyer,
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
                ['buyer_id', 'l3_id', 'year_id', 'month_id', 'volume', 'predict_smoothing_seas',
                 'predict_holt_seas', 'predict_arima', 'sellin_corr', 'predict_smoothing_corr_seas',
                 'predict_holt_corr_seas', 'predict_arima_corr', 'date_upload', 'status_id']]

            print(best_params_X)
            print("_______")
            print(best_params_Y)
            print(define_best_model_test(best_params_X, best_params_Y))

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

            df_final.to_csv('atyashevo_promo_23102023.csv', mode='a', decimal=',', sep=';')








main_prediction_3(datetime.now(), status_name=1)
