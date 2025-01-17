import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

import statistics
import pyodbc
from sqlalchemy import create_engine
import urllib

import math
from datetime import datetime

import warnings

warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Current version of project
def current_version():
    return "Forecast_3_models.ver_1.34"


# Function for defining quantile range to find outliers
def quantile_range(data: pd.Series):
    q25 = data.quantile(q=0.25)
    q75 = data.quantile(q=0.75)
    delta = q75 - q25
    boundaries = (q25 - 1.5 * delta, q75 + 1.5 * delta)
    return boundaries


# Function for defining 3 sigma range  to find outliers
def three_sigma_borders(data: pd.Series):
    low = data.mean() - 3 * data.std()
    high = data.mean() + 3 * data.std()
    res = (low, high)
    return res


# Function for defining seasonal coefficients for prediction in months
def seasonal(df: pd.DataFrame,  # DataFrame with (year, volume and month_id) columns
             n=12):  # Number of months
    full_year = []
    seas_dict_month = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0}  #
    for y in df.year.unique():  # research dataset for finding all full year samples (years with volume for n months
        if len(df[df.year == y]) == n:
            full_year.append(y)
    for f_y in full_year:  # calculation seasonal coefficients
        for m in range(1, 13):
            seas_dict_month[m] = seas_dict_month[m] + df[(df['year'] == f_y) & (df['month_id'] == m)]['volume'].values[
                0] / \
                                 df[df.year == f_y]['volume'].mean()
    if len(full_year) > 0:
        for x in seas_dict_month.keys():
            seas_dict_month[x] = seas_dict_month[x] / len(full_year)
    return seas_dict_month


# Function for defining datasets without sales last N months
def no_sales_criteria(df: pd.DataFrame,
                      n: int):
    if n == 0:
        print("No criteria about no sales period!")
        return False
    elif df.iloc[-n:].values.sum() <= 0.:
        print(f"Last {n} values is 0. Stop modeling")
        return True
    else:
        return False


# Build-up Calendar DataFrame (year, month_id) between start and end points
def sample_calendar(start_year: int,
                    start_month: int,
                    final_year: int,
                    final_month: int):
    res = []
    s = 1
    f = 12
    for y in range(start_year, final_year + 1):
        if y == start_year:
            s = start_month
        if y == final_year:
            f = final_month
        for x in range(s, f + 1):
            res.append([str(y) + '_' + str(x)])
            s = 1
            f = 12
    calendar = pd.DataFrame(res)
    return calendar


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


# Function for calculation quality metrics (BAES and MAPE)
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


# Build-up Simple Smoothing models
def SimpleSmooth_Seas(train,  # Train dataset
                      test,  # Test dataset
                      seas,  # Seasonal dataset
                      step,  # step for griding alpha parameter
                      df_modeling,  # Dataframe with representation of the process
                      modeling=0):  # parameter for switching mode of representation (1 - turn on. 0 - turn off)
    print('Start SimpleSmooth')
    # if we can't define DataFrame with seasonal coefficients --> Stop Simple Smoothing modeling
    for i in seas:
        if math.isnan(i) or i == 0:
            print("Seasonal is not defined")
            return {}
    error_dict_X_S = {}
    # Loop for checking models with set up alphas in range(0,1) with step
    for alpha in range(0, 100, int(step * 100)):
        try:
            threshold = 'No'
            model = SimpleExpSmoothing(train, initialization_method="heuristic").fit(
                smoothing_level=alpha / 100, optimized=False)
            fcast = model.forecast(len(test))
            fcast_seas = []
            for x, y in zip(fcast, seas):
                fcast_seas.append(x * y)
            # Check std for prediction on Train and Test. If std is near with train/test target std, we will add metrics
            # to dictionary with model_name and score for train and test
            if (statistics.pstdev(fcast_seas) / statistics.mean(fcast_seas) >=
                statistics.pstdev(test) / statistics.mean(test) * (1 - 0.5)) and \
                    (statistics.pstdev(model.fittedvalues) / statistics.mean(model.fittedvalues) >=
                     statistics.pstdev(train) / statistics.mean(train) * (1 - 0.5)):
                threshold = 'Added'
                error_dict_X_S['Smoothing_' + str(alpha / 100)] = [
                    mean_absolute_error(train, model.fittedvalues),
                    mean_absolute_error(test, fcast_seas)]
            # Build-up Dataframe with representation of the process
            if modeling == 1:
                res_modeling = np.append(model.fittedvalues, fcast_seas)
                df_modeling_1 = df_modeling.copy()
                df_modeling_1 = df_modeling_1.assign(prediction=res_modeling)
                df_modeling_1 = df_modeling_1.assign(parameters='Smoothing_' + str(alpha / 100))
                df_modeling_1 = df_modeling_1.assign(train_error=mean_absolute_error(train, model.fittedvalues))
                df_modeling_1 = df_modeling_1.assign(test_error=mean_absolute_error(test, fcast_seas))
                df_modeling_1 = df_modeling_1.assign(threshold=threshold)
                df_modeling_1.to_csv("Modeling.csv", mode="a")
        except Exception as e:
            continue
        # Check model with estimated method of initialization
        try:
            threshold = 'No'
            model = SimpleExpSmoothing(train, initialization_method="estimated").fit()
            fcast = model.forecast(len(test))
            fcast_seas = []
            for x, y in zip(fcast, seas):
                fcast_seas.append(x * y)
            # Check std
            if (statistics.pstdev(fcast_seas) / statistics.mean(fcast_seas) >=
                statistics.pstdev(test) / statistics.mean(test) * (1 - 0.5)) and \
                    (statistics.pstdev(model.fittedvalues) / statistics.mean(model.fittedvalues) >=
                     statistics.pstdev(train) / statistics.mean(train) * (1 - 0.5)):
                threshold = 'Added'
                error_dict_X_S['Smoothing_' + str(model.model.params["smoothing_level"])] = [
                    mean_absolute_error(train, model.fittedvalues),
                    mean_absolute_error(test, fcast_seas)]
            # Build-up Dataframe with representation of the process
            if modeling == 1:
                res_modeling = np.append(model.fittedvalues, fcast_seas)
                df_modeling_1 = df_modeling.copy()
                df_modeling_1 = df_modeling_1.assign(prediction=res_modeling)
                df_modeling_1 = df_modeling_1.assign(
                    parameters='Smoothing_' + str(model.model.params["smoothing_level"]))
                df_modeling_1 = df_modeling_1.assign(train_error=mean_absolute_error(train, model.fittedvalues))
                df_modeling_1 = df_modeling_1.assign(test_error=mean_absolute_error(test, fcast_seas))
                df_modeling_1 = df_modeling_1.assign(threshold=threshold)
                df_modeling_1.to_csv("Modeling.csv", mode="a")
        except Exception as e:
            continue
    # return dictionary {model_name: mae_score}
    return error_dict_X_S


# # Build-up Holt method models
def Holt_Seas(train,  # Train dataset
              test,  # Test dataset
              seas,  # Seasonal dataset
              step1,  # step for griding alpha parameter
              step2,  # step for griding beta parameter
              df_modeling,  # Dataframe with representation of the process
              modeling=0):  # parameter for switching mode of representation (1 - turn on. 0 - turn off)

    print('Start Holt')
    # if we can't define DataFrame with seasonal coefficients --> Stop Holt method modeling
    for i in seas:
        if math.isnan(i) or i == 0:
            print("Seasonal is not defined")
            return {}
    error_dict_X_S = {}
    # Loop for checking models with set up alphas and betas in range(0,1) with step1 and step2
    for alpha in range(0, 100, int(step1 * 100)):
        for beta in range(0, 100, int(step2 * 100)):
            for exp in [True, False]:
                for damp in [True, False]:
                    try:
                        threshold = 'No'
                        model = Holt(train, exponential=exp, damped_trend=damp, initialization_method="estimated").fit(
                            smoothing_level=alpha / 100, smoothing_trend=beta / 100)
                        fcast = model.forecast(len(test))
                        fcast_seas = []
                        for x, y in zip(fcast, seas):
                            fcast_seas.append(x * y)
                        # Check std for prediction on Train and Test. If std is near with train/test target std,
                        # we will add metrics to dictionary with model_name and score for train and test
                        if (statistics.pstdev(fcast_seas) / statistics.mean(fcast_seas) >=
                            statistics.pstdev(test) / statistics.mean(test) * (1 - 0.5)) and \
                                (statistics.pstdev(model.fittedvalues) / statistics.mean(model.fittedvalues) >=
                                 statistics.pstdev(train) / statistics.mean(train) * (1 - 0.5)):
                            threshold = 'Added'
                            error_dict_X_S['Holt_' + str(exp) + '/' + str(damp) + '/' + str(alpha / 100) + '/' + str(
                                beta / 100)] = [
                                mean_absolute_error(train, model.fittedvalues),
                                mean_absolute_error(test, fcast_seas)]

                        # Build-up Dataframe with representation of the process
                        if modeling == 1:
                            res_modeling = np.append(model.fittedvalues, fcast_seas)
                            df_modeling_1 = df_modeling.copy()
                            df_modeling_1 = df_modeling_1.assign(prediction=res_modeling)
                            df_modeling_1 = df_modeling_1.assign(parameters='Holt_' + str(exp) + '/' + str(damp) + '/' +
                                                                            str(alpha / 100) + '/' + str(beta / 100))
                            df_modeling_1 = df_modeling_1.assign(
                                train_error=mean_absolute_error(train, model.fittedvalues))
                            df_modeling_1 = df_modeling_1.assign(test_error=mean_absolute_error(test, fcast_seas))
                            df_modeling_1 = df_modeling_1.assign(threshold=threshold)
                            df_modeling_1.to_csv("Modeling.csv", mode="a")
                    except Exception as e:
                        continue
    # return dictionary {model_name: mae_score}
    return error_dict_X_S


# Build-up Holt-Winters method models
def Holt_Winters(train,  # Train dataset
                 test,  # Test dataset
                 step1,  # step for griding alpha parameter
                 step2,  # step for griding beta parameter
                 step3,  # step for griding gamma parameter
                 df_modeling,  # Dataframe with representation of the process
                 modeling=0,  # parameter for switching mode of representation (1 - turn on. 0 - turn off)
                 seasonal_period=12):  # seasonal period
    print('Start Holt_Winters')
    error_dict_X_S = {}
    # Loop for checking models with set up alphas, betas and gammas in range(0,1) with step1, step2, step3
    for alpha in range(0, 100, int(step1 * 100)):
        for beta in range(0, 100, int(step2 * 100)):
            for gamma in range(0, 100, int(step3 * 100)):
                for tr in ['add', 'mul', None]:
                    for damp in [True, False]:
                        for ss in ['add', 'mul']:
                            try:
                                threshold = 'No'
                                model = ExponentialSmoothing(train,
                                                             seasonal_periods=seasonal_period,
                                                             trend=tr,
                                                             damped_trend=damp,
                                                             seasonal=ss,
                                                             initialization_method='estimated'
                                                             ).fit(smoothing_level=alpha / 100,
                                                                   smoothing_trend=beta / 100,
                                                                   smoothing_seasonal=gamma / 100)
                                fcast = model.forecast(len(test))
                                # Check std for prediction on Train and Test. If std is near with train/test target std,
                                # we will add metrics to dictionary with model_name and score for train and test
                                if (statistics.pstdev(fcast) / statistics.mean(fcast) >=
                                    statistics.pstdev(test) / statistics.mean(test) * (1 - 0.5)) and \
                                        (statistics.pstdev(model.fittedvalues) / statistics.mean(model.fittedvalues) >=
                                         statistics.pstdev(train) / statistics.mean(train) * (1 - 0.5)):
                                    threshold = 'Added'
                                    error_dict_X_S['Holt-Winters_' + str(alpha / 100) + '/' + str(beta / 100) + '/'
                                                   + str(gamma / 100) + '/' + str(tr) + '/' + str(damp) +
                                                   '/' + str(ss)] = [mean_absolute_error(train, model.fittedvalues),
                                                                     mean_absolute_error(test, fcast)]
                                # Build-up Dataframe with representation of the process
                                if modeling == 1:
                                    res_modeling = np.append(model.fittedvalues, fcast)
                                    df_modeling_1 = df_modeling.copy()
                                    df_modeling_1 = df_modeling_1.assign(prediction=res_modeling)
                                    df_modeling_1 = df_modeling_1.assign(
                                        parameters='Holt-Winters_' + str(alpha / 100) + '/' + str(beta / 100) + '/'
                                                   + str(gamma / 100) + '/' + str(tr) + '/' + str(damp) +
                                                   '/' + str(ss))
                                    df_modeling_1 = df_modeling_1.assign(
                                        train_error=mean_absolute_error(train, model.fittedvalues))
                                    df_modeling_1 = df_modeling_1.assign(test_error=mean_absolute_error(test, fcast))
                                    df_modeling_1 = df_modeling_1.assign(threshold=threshold)
                                    df_modeling_1.to_csv("Modeling.csv", mode="a")

                            except Exception as e:
                                continue
    # return dictionary {model_name: mae_score}
    return error_dict_X_S


# # Build-up ARIMA method models
def Arima(train,  # Train dataset
          test,  # Test dataset
          p_max,  # step for griding p parameter
          q_max,  # step for griding q parameter
          d_max,  # step for griding d parameter
          df_modeling,  # Dataframe with representation of the process
          modeling=0):  # parameter for switching mode of representation (1 - turn on. 0 - turn off)

    print('Start Arima')
    error_dict_X_S = {}
    # Loop for checking models with set up p, q and d in range(0,max) with p_max, q_max, d_max
    for d in range(1, d_max + 1):
        for q in range(1, q_max + 1):
            for p in range(1, p_max + 1, 1):
                try:
                    threshold = 'No'
                    model = ARIMA(train, order=(p, q, d)).fit()
                    fcast = model.forecast(len(test))
                    # Check std for prediction on Train and Test. If std is near with train/test target std,
                    # we will add metrics to dictionary with model_name and score for train and test
                    if (statistics.pstdev(fcast) / statistics.mean(fcast) >=
                        statistics.pstdev(test) / statistics.mean(test) * (1 - 0.5)) and \
                            (statistics.pstdev(model.fittedvalues) / statistics.mean(model.fittedvalues) >=
                             statistics.pstdev(train) / statistics.mean(train) * (1 - 0.5)):
                        threshold = 'Added'
                        error_dict_X_S['ARIMA_' + str(p) + '/' + str(q) + '/' + str(d)] = [
                            mean_absolute_error(train, model.fittedvalues),
                            mean_absolute_error(test, fcast)]
                    if modeling == 1:
                        res_modeling = np.append(model.fittedvalues, fcast)
                        df_modeling_1 = df_modeling.copy()
                        df_modeling_1 = df_modeling_1.assign(prediction=res_modeling)
                        df_modeling_1 = df_modeling_1.assign(
                            parameters='ARIMA_' + str(p) + '/' + str(q) + '/' + str(d))
                        df_modeling_1 = df_modeling_1.assign(
                            train_error=mean_absolute_error(train, model.fittedvalues))
                        df_modeling_1 = df_modeling_1.assign(test_error=mean_absolute_error(test, fcast))
                        df_modeling_1 = df_modeling_1.assign(threshold=threshold)
                        df_modeling_1.to_csv("Modeling.csv", mode="a")

                except Exception as e:
                    continue

    # return dictionary {model_name: mae_score}
    return error_dict_X_S


# Finding best_params from error dictionary
def best_params_founder(err_dict,  # Error dictionary
                        best_params):  # List of the best parameters for every type of models
    if len(err_dict) > 0:
        init = sorted([x[0] for x in err_dict.values()])
        threshold = 0.25
        best_train_err = init[0] * (1 + threshold)
        # Initialize list with only good score for train (not worse than % of best train score)
        init_1 = sorted([x[1] for x in err_dict.values() if x[0] <= best_train_err])
        for keys, values in err_dict.items():
            if (values[1] == init_1[0]) and (keys not in best_params):
                add_str = keys + "_" + str(init_1[0])
                best_params.append(add_str)
                break
    return best_params


def best_models_fit(df,  # Full dataset for prediction
                    best_params,  # List with parameters of best models each type
                    period,  # How many predictions we have to make
                    seasonal_period=12):  # # seasonal period for Holt-Winters model
    res1 = []
    models = ['Smoothing', 'Holt', 'Holt-Winters', 'ARIMA']
    best_params_cor = []
    # Build-up full list of models
    if len(best_params) < 4:
        for elem in models:
            d = len(best_params_cor)
            for x in best_params:
                if elem == x.split("_")[0]:
                    best_params_cor.append(x)
            if d == len(best_params_cor):
                best_params_cor.append('None_None')
        best_params = best_params_cor

    #  If the first element id Smoothing model's parameters, initialize it. Else add 0 numbers
    if best_params[0].split("_")[0] == 'Smoothing':
        model_params = best_params[0].split("_")[1]
        model_smoothing = SimpleExpSmoothing(df, initialization_method="heuristic").fit(
            smoothing_level=float(model_params), optimized=False)
        fcast_smoothing = model_smoothing.forecast(period)
        res1.append([0] + list(model_smoothing.fittedvalues) + list(fcast_smoothing))
    else:
        print('No Smoothing model')
        res1.append([0 for x in range(len(df) + period + 1)])

    #  If the second element id Holt model's parameters, initialize it. Else add 0 numbers
    if best_params[1].split("_")[0] == 'Holt':
        model_params = best_params[1].split("_")[1]
        model_holt = Holt(df, exponential=(False if model_params.split('/')[0] == 'False'
                                           else model_params.split('/')[0]),
                          damped_trend=(False if model_params.split('/')[1] == 'False'
                                        else model_params.split('/')[1]),
                          initialization_method="estimated").fit(
            smoothing_level=float(model_params.split('/')[2]),
            smoothing_trend=float(model_params.split('/')[3]))
        fcast_holt = model_holt.forecast(period)
        res1.append([0] + list(model_holt.fittedvalues) + list(fcast_holt))
    else:
        print('No Holt model')
        res1.append([0 for x in range(len(df) + period + 1)])

    #  If the forth element id ARIMA model's parameters, initialize it. Else add 0 numbers
    if best_params[3].split("_")[0] == 'ARIMA':
        model_params = best_params[3].split("_")[1]
        try:
            model_arima = ARIMA(df, order=(
                int(model_params.split('/')[0]),
                int(model_params.split('/')[1]),
                int(model_params.split('/')[2]))).fit()
            fcast_arima = model_arima.forecast(period + 1)
            res1.append(list(model_arima.fittedvalues) + list(fcast_arima))
        except np.linalg.LinAlgError:
            print("Something is not good")
            res1.append([0 for x in range(len(df) + period + 1)])
        except IndexError:
            print("Something is not good")
            res1.append([0 for x in range(len(df) + period + 1)])
    else:
        print('No ARIMA model')
        res1.append([0 for x in range(len(df) + period + 1)])

    #  If the third element id ARIMA model's parameters, initialize it. Else add 0 numbers
    if best_params[2].split("_")[0] == 'Holt-Winters':
        model_params = best_params[2].split("_")[1]
        model_holt_winters = ExponentialSmoothing(df,
                                                  seasonal_periods=seasonal_period,
                                                  trend=(None if model_params.split('/')[3] == 'None'
                                                         else model_params.split('/')[3]),
                                                  damped_trend=(False if model_params.split('/')[4] == 'False'
                                                                else model_params.split('/')[4]),
                                                  seasonal=(None if model_params.split('/')[5] == 'None'
                                                            else model_params.split('/')[5]),
                                                  initialization_method='estimated'
                                                  ).fit(smoothing_level=float(model_params.split('/')[0]),
                                                        smoothing_trend=float(model_params.split('/')[1]),
                                                        smoothing_seasonal=float(model_params.split('/')[2]))
        fcast_holt_winters = model_holt_winters.forecast(period + 1)
        res1.append(list(model_holt_winters.fittedvalues) + list(fcast_holt_winters))
    else:
        print('No Holt-Winters models')
        res1.append([0 for x in range(len(df) + period + 1)])

    return res1


# Going through best parameters lists to define the best model on test score v.1
def define_best_model_test(best_params_X,  # The first sample of model's parameters and scores list
                           best_params_Y):  # The second sample of model's parameters and scores list
    best_score = -1
    best_model = "Unknown"

    for elem in best_params_X:
        model_name = elem.split("_")[0]
        model_score = float(elem.split("_")[2])
        if best_score < 0 or best_score > model_score:
            best_score = model_score
            best_model = model_name
            # print(best_model, best_score)
    for elem in best_params_Y:
        model_name = elem.split("_")[0]
        model_score = float(elem.split("_")[2])
        if best_score < 0 or best_score > model_score:
            best_score = model_score
            best_model = model_name + "_corr"
            # print(best_model, best_score)
    # return the best model's name
    return best_model


# Going through best parameters lists to define the best model on test score v.2
def define_best_model_test_np(best_params_X,  # The first sample of model's parameters and scores list
                              best_params_Y):  # The second sample of model's parameters and scores list
    x = []
    for elem in best_params_X:
        x.append([elem.split("_")[0], float(elem.split("_")[2])])
    for elem in best_params_Y:
        x.append([elem.split("_")[0] + "_corr", float(elem.split("_")[2])])
    np1 = np.array(x, dtype=object)
    if len(np1) > 1:
        np1 = np1[np1[:, 1].argsort()]
    # return the best model's name
    return np1


# Connect to SQL database for extracting necessary samples of data for prediction loop
def connection_DB(time_connection,  # Start time
                  chain_list,  # List of necessary buyers
                  category_list,  # List of necessary categories of goods
                  status_name):  # Type of sales (if status_name=0, we will download all type of sales)
    CONNECTION_PATH = Path()
    # Connection parameters are inside txt file
    FILENAME = "connection_DB.txt"
    CONNECTION_FILENAME = CONNECTION_PATH / FILENAME
    with open(CONNECTION_FILENAME) as f:
        lines = f.readlines()
    connect_str = ""
    for x in lines:
        connect_str += x.replace('/n', '')
    connect_str = " ".join(connect_str.split())
    print('Start', time_connection)

    connection = pyodbc.connect(connect_str)

    print('Time of connection', datetime.now() - time_connection)

    # Extracting full list of buyers and categories of goods in case when we will predict full data
    chain_fulllist = pd.read_sql("SELECT [id] FROM [MVPPresident].[dbo].[_spr_address_cpg]",
                                 connection)

    category_fulllist = pd.read_sql("SELECT [id] FROM [MVPPresident].[dbo].[_spr_sku_ppg]",
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
        data = pd.read_sql(f"SELECT * FROM [dbo].[SalesInWeek] "
                           f"WHERE cpg_id in {chain_str} and ppg_id in {category_str}",
                           connection)
    elif status_name in (1, 2):
        data = pd.read_sql(f"SELECT * FROM [dbo].[SalesInWeek] "
                           f"WHERE cpg_id in {chain_str} and ppg_id in {category_str}"
                           f"and status_id = {status_name}",
                           connection)
    else:
        print("Incorrect status_id")
        data = pd.DataFrame()

    print(f"Was extracted {len(data)} string")

    # Extracting table with years and id
    year_calendar = pd.read_sql("SELECT [id], [year] FROM [MVPPresident].[dbo].[_spr_date_year]",
                                connection)

    year_calendar = year_calendar.astype({'id': np.int64, 'year': np.int64})

    print('Time of download', datetime.now() - time_connection)

    connection.close()

    return data, year_calendar


# Uploading predictions in SQL database
def download_DB(df):
    CONNECTION_PATH = Path()
    FILENAME = "connection_DB.txt"
    CONNECTION_FILENAME = CONNECTION_PATH / FILENAME
    with open(CONNECTION_FILENAME) as f:
        lines = f.readlines()
    connect_str = ""
    for x in lines:
        connect_str += x.replace('/n', '')
    connect_str = " ".join(connect_str.split())
    connection = pyodbc.connect(connect_str)
    cursor = connection.cursor()

    quoted = urllib.parse.quote_plus(connect_str)
    engine = create_engine(f'mssql+pyodbc:///?odbc_connect={quoted}')

    with engine.connect() as cnn:
        print("~~~Downloading~~~")
        df.to_sql('forecast_g_raw', schema='dbo', con=cnn, if_exists='append', index=False)


# Main prediction loop
def main_prediction(chain_list,  # List of necessary buyers
                    category_list,  # List of necessary categories of goods
                    time_connection,  # Start time
                    final_date,  # Final month for fact data
                    skip_months,  # Amount of monthes to skip
                    period,  # Period of forecasting
                    status_name=0,  # Type of sales (if status_name=0, we will download all type of sales)
                    sales_criteria=6,
                    download_flag=1):
    # Extracting data from SQL database
    data, year_calendar = connection_DB(time_connection,
                                        chain_list,
                                        category_list,
                                        status_name)
    if status_name == 0:
        status_name_act = 3
    else:
        status_name_act = status_name

    #  Some data preparation
    df_new_1 = data.loc[(data['chain_name'].notna()) | (data['l3_name'].notna())]
    df_new_1 = df_new_1[['year', 'year_id', 'month_id', 'volume', 'cpg_id', 'ppg_id', 'cpg_name', 'ppg_name']]
    df_new_1 = df_new_1.rename(columns={'ppg_id': 'l3_id'})
    chain_name = df_new_1.cpg_id.value_counts().index.to_list()
    print(chain_name)

    for chain in chain_name:
        # Build-up empty DataFrame for our predictions
        df_final_full = pd.DataFrame(columns=['l3_id', 'buyer_id', 'year_id', 'month_id', 'volume',
                                              'predict_smoothing_seas', 'predict_holt_seas',
                                              'predict_arima', 'predict_holt_wint', 'sellin_corr',
                                              'predict_smoothing_corr_seas', 'predict_holt_corr_seas',
                                              'predict_arima_corr', 'predict_holt_wint_corr',
                                              'date_upload', 'best_model_total', 'best_model', 'status_id',
                                              'best_model_value', 'predict_unknown'])
        # Extracting categories of goods list for current buyer
        l3_name = df_new_1[df_new_1.cpg_id == chain].l3_id.value_counts().index.to_list()
        print(l3_name)
        for l3 in l3_name:
            print(f'Start prediction {chain} for group {l3}')

            # Some data preparation
            df_new_1_2 = df_new_1[(df_new_1.cpg_id == chain) & (df_new_1.l3_id == l3)]
            df_new_1_2 = df_new_1_2[['year', 'month_id', 'volume']]
            df_new_1_2 = df_new_1_2.groupby(by=['year', 'month_id'], as_index=False).sum()
            df_new_1_2 = df_new_1_2.astype({'year': np.int64, 'month_id': np.int64, 'volume': np.float64})

            # If data is not enough for prediction, we go next
            # if len(df_new_1_2) < 4:
            #     print("Not enough length of dataset - ", len(df_new_1_2))
            #     continue
            date_object = datetime.strptime(final_date, "%Y-%m-%d")

            # Извлекаем год и месяц
            final_date_year = date_object.year
            final_date_month = date_object.month

            # Extrapolate our data on full year/month period with small values on months without sale
            df_new_1_2['Cal'] = df_new_1_2.apply(lambda var: str(int(var.year)) + '_' + str(int(var.month_id)), axis=1)
            cals = sample_calendar(df_new_1_2.year.min(),
                                   df_new_1_2[df_new_1_2.year == df_new_1_2.year.min()].month_id.min(),
                                   final_date_year,
                                   final_date_month)
            df_new_1_2 = df_new_1_2.merge(cals, left_on='Cal', right_on=0, how='right')
            df_new_1_2['year'] = df_new_1_2.apply(lambda var: int(var.Cal.split('_')[0]), axis=1)
            df_new_1_2['month_id'] = df_new_1_2.apply(lambda var: int(var.Cal.split('_')[1]), axis=1)
            df_new_1_2['volume'] = df_new_1_2['volume'].fillna(0)
            df_new_1_2 = df_new_1_2.drop(['Cal', 0], axis=1)

            df_new_1_2['Seas'] = df_new_1_2['month_id'].map(seasonal(df_new_1_2, 12))

            df_new_1_2 = df_new_1_2.sort_values(by=['year', 'month_id'], ascending=True)
            # If we don't have sales last N moths, we go next
            if no_sales_criteria(df_new_1_2.volume, sales_criteria):
                print(f"No sales criteria cpg_id - {chain} and l3_id - {l3}")
                continue

            df_new_1_2.loc[(df_new_1_2['volume'] <= 0), 'volume'] = 0.000001

            # Making Series for predictions
            X = df_new_1_2['volume']
            X = X.reset_index(drop=True)

            # Copy our data and make corrections with outliers
            Y = df_new_1_2['volume'].copy()
            quan_res = quantile_range(Y)
            # sigma_res = three_sigma_borders(Y)
            Y[(Y.values < quan_res[0])] = quan_res[0]
            Y[(Y.values > quan_res[1])] = quan_res[1]

            t = skip_months  # except 1 month from dataset, cuz it can be not full else
            X_m = X[:-t]
            train_size = int(len(X_m) * 0.75)
            train_X, test_X = X_m[:train_size].to_list(), X_m[train_size:].to_list()

            Y_m = Y[:-t]
            train_Y, test_Y = Y_m[:train_size].to_list(), Y_m[train_size:].to_list()

            Seas = df_new_1_2['Seas'].reset_index(drop=True).to_list()[-len(test_X) - t:-t]

            best_params_X = []
            best_params_Y = []

            # Preparation for representation modeling, if it will be needed
            df_modeling = df_new_1_2.assign(set_type='Train')
            df_modeling.iloc[train_size:, 4] = 'Test'
            df_modeling = df_modeling.drop(['Seas'], axis=1)
            df_modeling = df_modeling.assign(cpg=chain)
            df_modeling = df_modeling.assign(l3=l3)
            df_modeling_X = df_modeling.assign(correction='No')
            df_modeling_Y = df_modeling.assign(correction='Yes')

            print("~~~~~prediction for X~~~~~")
            best_params_founder(
                SimpleSmooth_Seas(train_X, test_X, Seas, 0.05, df_modeling_X.iloc[:-t]),
                best_params_X)
            best_params_founder(
                Holt_Seas(train_X, test_X, Seas, 0.1, 0.1, df_modeling_X.iloc[:-t]),
                best_params_X)
            best_params_founder(
                Holt_Winters(train_X, test_X, 0.2, 0.2, 0.2, df_modeling_X.iloc[:-t]),
                best_params_X)
            best_params_founder(
                Arima(train_X, test_X, 12, 2, 2, df_modeling_X.iloc[:-t]),
                best_params_X)

            # If Y and X is equal we will not start prediction loop for Y.
            # And set up best_models os equal best_models fro X
            if X_m.equals(Y_m):
                print("No correction of dataset")
                best_params_Y = best_params_X
            else:
                print("~~~~~prediction for Y~~~~~")
                best_params_founder(
                    SimpleSmooth_Seas(train_Y, test_Y, Seas, 0.05, df_modeling_Y.iloc[:-t]),
                    best_params_Y)
                best_params_founder(
                    Holt_Seas(train_Y, test_Y, Seas, 0.1, 0.1, df_modeling_Y.iloc[:-t]),
                    best_params_Y)
                best_params_founder(
                    Holt_Winters(train_Y, test_Y, 0.2, 0.2, 0.2, df_modeling_Y.iloc[:-t]),
                    best_params_Y)
                best_params_founder(
                    Arima(train_Y, test_Y, 12, 2, 2, df_modeling_Y.iloc[:-t]),
                    best_params_Y)

            print(f"Best models {chain} and {l3}: ", best_params_X)
            print(f"Best models {chain} and {l3} corr: ", best_params_Y)

            # period = 14
            df_final = df_new_1_2.copy()
            df_final.drop('Seas', axis=1)

            res_X = best_models_fit(X_m, best_params_X, period)

            if X_m.equals(Y_m):
                res_Y = res_X
            else:
                res_Y = best_models_fit(Y_m, best_params_Y, period)

            # Build-up fact part of prediction dataset
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
            df_final = df_final.assign(date_upload=time_connection)
            df_final = df_final.assign(status_id=status_name_act)

            # Build-up prediction part of prediction dataset
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
                                     "predict_holt_wint": res_X[3][len_st + i],
                                     "l3_id": l3,
                                     "buyer_id": chain,
                                     "region_id": '',
                                     "sellin_corr": 0,
                                     "predict_smoothing_corr": res_Y[0][len_st + i],
                                     "predict_holt_corr": res_Y[1][len_st + i],
                                     "predict_arima_corr": res_Y[2][len_st + i],
                                     "predict_holt_wint_corr": res_Y[3][len_st + i],
                                     "date_upload": time_connection,
                                     "status_id": status_name_act})
                df_final = df_final.append(new_row, ignore_index=True)

            df_final['Seas'] = df_final['month_id'].map(seasonal(df_new_1_2, 12))

            # Apply seasonal coefficients for Smoothing and Holt model's predictions
            df_final['predict_smoothing_seas'] = df_final['Seas'] * df_final['predict_smoothing']
            df_final['predict_holt_seas'] = df_final['Seas'] * df_final['predict_holt']

            df_final['predict_smoothing_corr_seas'] = df_final['Seas'] * df_final['predict_smoothing_corr']
            df_final['predict_holt_corr_seas'] = df_final['Seas'] * df_final['predict_holt_corr']

            # Basic correction for prediction values
            for column in ['predict_smoothing', 'predict_holt', 'predict_arima',
                           'predict_smoothing_corr', 'predict_holt_corr', 'predict_arima_corr',
                           'predict_smoothing_seas', 'predict_holt_seas', 'predict_smoothing_corr_seas',
                           'predict_holt_corr_seas', 'predict_holt_wint', 'predict_holt_wint_corr']:
                df_final[column][(df_final[column].values < 0)] = 0
                df_final[column][(df_final[column].values > 1000000000000)] = 1000000000000

            df_final = df_final.fillna(0)
            df_final = df_final.merge(year_calendar, left_on='year', right_on='year', how='left')
            df_final = df_final.rename(columns={'id': 'year_id'})
            df_final = df_final.drop(
                ['predict_smoothing', 'predict_holt', 'predict_smoothing_corr', 'predict_holt_corr'], axis=1)
            df_final = df_final[
                ['buyer_id', 'l3_id', 'year_id', 'month_id', 'volume', 'predict_smoothing_seas',
                 'predict_holt_seas', 'predict_arima', 'predict_holt_wint', 'sellin_corr',
                 'predict_smoothing_corr_seas', 'predict_holt_corr_seas', 'predict_holt_wint_corr',
                 'predict_arima_corr', 'date_upload', 'status_id']]

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

            # Defining best model on test with some hand-made updates to business task
            # Growth next year has to be hand limited (now it is not higher than 1.8 and not lower 0.4 to prev year
            last_year_index = max(len_st - t - 12, 0)

            last_year_volume = df_final[last_year_index:len_st - t].volume.sum()

            best_model_df_1 = 'Unknown'

            len_of_last_year = len(df_final[last_year_index:len_st - t])
            for bm in define_best_model_test_np(best_params_X, best_params_Y):
                model_ = model_dict[bm[0]]
                next_year_volume = df_final[len_st - t:len_st - t + len_of_last_year][model_].sum()
                print(f"Growth with model {model_} is {next_year_volume / last_year_volume}")
                if ((next_year_volume / last_year_volume > 1.8 or next_year_volume / last_year_volume < 0.6) or
                        next_year_volume == 0):
                    continue
                else:
                    best_model_df_1 = model_
                    break

            print("Best model test updated", best_model_df_1)
            # Defining best model on whole dataset
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

            df_final['best_model_total'] = best_model
            df_final['best_model'] = best_model_df_1
            df_final['predict_unknown'] = 0
            unknown_index = max(len_st - t - 6, 0)

            if best_model_df_1 == 'Unknown':
                last_year_mean = df_final[unknown_index:len_st - t].volume.mean()
                df_final['best_model_value'] = last_year_mean
                df_final['predict_unknown'] = last_year_mean
            else:
                df_final['best_model_value'] = df_final[best_model_df_1]

            df_final_full = df_final_full.append(df_final)
        print(df_final_full)

        df_final_full = df_final_full.astype({'l3_id': np.int64,
                                              'buyer_id': np.int64,
                                              'year_id': np.int64,
                                              'month_id': np.int64,
                                              'volume': np.float64,
                                              'predict_smoothing_seas': np.float64,
                                              'predict_holt_seas': np.float64,
                                              'predict_arima': np.float64,
                                              'predict_holt_wint': np.float64,
                                              'sellin_corr': np.float64,
                                              'predict_smoothing_corr_seas': np.float64,
                                              'predict_holt_corr_seas': np.float64,
                                              'predict_arima_corr': np.float64,
                                              'predict_holt_wint_corr': np.float64,
                                              'date_upload': np.datetime64,
                                              'best_model_total': np.str_,
                                              'best_model': np.str_,
                                              'status_id': np.int64,
                                              'best_model_value': np.float64,
                                              'predict_unknown': np.float64})

        df_final_full_reorder = df_final_full[['buyer_id',
                                               'l3_id',
                                               'year_id',
                                               'month_id',
                                               'volume',
                                               'predict_smoothing_seas',
                                               'predict_holt_seas',
                                               'predict_arima',
                                               'sellin_corr',
                                               'predict_smoothing_corr_seas',
                                               'predict_holt_corr_seas',
                                               'predict_arima_corr',
                                               'date_upload',
                                               'best_model',
                                               'status_id',
                                               'best_model_total',
                                               'best_model_value',
                                               'predict_holt_wint',
                                               'predict_holt_wint_corr',
                                               'predict_unknown'
                                               ]]

        # Write result to the file
        filename = time_connection.strftime("%d%m%y")
        file_tag = 'result'
        filename += "___" + str(file_tag) + ".csv"
        filename = "data/" + filename
        print(filename)
        df_final_full_reorder.to_csv(filename, decimal=',', index=False)

        # Downloading result in SQL database
        if len(df_final_full_reorder) > 0 and download_flag == 1:
            # continue
            download_DB(df_final_full_reorder)
            print(f'Download {chain} was successful')
            print('Time of download full category prediction', datetime.now() - time_connection)

        # Turning on SQL procedure
        # sql_split_forecast = "EXEC split_forecast_g @DateToLoad=?"
        # params = (time_connection)
        # cursor.execute(sql_split_forecast, params)
        print('Full time', datetime.now() - time_connection)


# Developing of prediction loop for weeks dataset (in process now)
def main_prediction_week(df, coef_smoothing, filling_calendar='yes', start_period=0, end_period=datetime.now()):
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


# Press the green button in the gutter to run the script.
#
# [7230,7236,7238,7241,7242,7245,7246,7256,7258,7259,7261,7265,7267,7268,7269,7270,7272,7273]
if __name__ == '__main__':
    date_time_obj = datetime.strptime('2024-10-17 00:00:00.000', '%Y-%m-%d %H:%M:%S.%f')
    date_time_obj = datetime.now()
    main_prediction(
        chain_list=[7011],
        category_list=[189],
        time_connection=date_time_obj,
        final_date='2024-11-01',
        skip_months=1,
        period=13,
        status_name=0,
        sales_criteria=6,
        download_flag=0)
