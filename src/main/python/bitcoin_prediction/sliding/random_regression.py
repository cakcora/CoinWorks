from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np, tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os
import csv
import gc
from sklearn.metrics import mean_squared_error
import math
from os.path import dirname as up
from sklearn.metrics import mean_squared_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



ALL_YEAR_INPUT_ALLOWED = False
START_YEAR = 2015
END_YEAR = 2016
SLIDING_BATCH_SIZE = 60


PRICED_BITCOIN_FILE_NAME = "pricedBitcoin2009-2018.csv"
DAILY_OCCURRENCE_FILE_NAME = "dailyOccmatrices2009-2018\\dailyOccmatrices\\"
LOG_FILE = 'C:\\Users\\nca150130\\PycharmProjects\\CoinWorks\\src\\main\\python\\results\\rmse\\random_regression\\bitcoin_prices_with_chainlet_' + str(START_YEAR) + "_" + str(END_YEAR) + ".csv"

CHAINLET_ALLOWED = True

ROW = -1
COLUMN = -1
TEST_SPLIT = 0.2


def merge_data(occurrence_data, daily_occurrence_normalized_matrix, aggregation_of_previous_days_allowed):
    if(aggregation_of_previous_days_allowed):
        if occurrence_data.size==0:
            occurrence_data = daily_occurrence_normalized_matrix
        else:
            occurrence_data = np.add(occurrence_data, daily_occurrence_normalized_matrix)
    else:
        if occurrence_data.size==0:
            occurrence_data = daily_occurrence_normalized_matrix
        else:
            occurrence_data = np.concatenate((occurrence_data, daily_occurrence_normalized_matrix), axis=1)
    return occurrence_data

def get_daily_occurrence_matrices(priced_bitcoin, current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    previous_price_data = np.array([], dtype=np.float32)
    occurrence_data = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])
            daily_occurrence_normalized_matrix = get_normalized_matrix_from_file(row['day'], row['year'], row['totaltx'])
            occurrence_data = merge_data(occurrence_data, daily_occurrence_normalized_matrix, aggregation_of_previous_days_allowed)
    if(is_price_of_previous_days_allowed):
        if(CHAINLET_ALLOWED):
            occurrence_data = np.concatenate((occurrence_data, np.asarray(previous_price_data).reshape(1,-1)), axis=1)
        else:
            occurrence_data = np.asarray(previous_price_data).reshape(1,-1)
    occurrence_input = np.concatenate((occurrence_data, np.asarray(current_row['price']).reshape(1,1)), axis=1)
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['day']).reshape(1,1)), axis=1)
    return occurrence_input

def get_normalized_matrix_from_file(day, year, totaltx):
    data_file_path = get_data_file_path()
    daily_occurrence_file_path = os.path.join(data_file_path, DAILY_OCCURRENCE_FILE_NAME, "occ" + str(year) + '{:03}'.format(day) + ".csv")
    daily_occurrence_matrix = pd.read_csv(daily_occurrence_file_path, sep=",", header=None).values
    return np.asarray(daily_occurrence_matrix).reshape(1, daily_occurrence_matrix.size)/totaltx

def get_data_file_path():
    return os.path.join(up(up(up(up(up(up(__file__)))))), "data")

def filter_data(priced_bitcoin):
    end_day_of_previous_year = max(priced_bitcoin[priced_bitcoin['year'] == START_YEAR-1]["day"].values)
    start_index_of_previous_year = end_day_of_previous_year - SLIDING_BATCH_SIZE - window_size
    previous_year_batch = priced_bitcoin[(priced_bitcoin['year'] == START_YEAR-1) & (priced_bitcoin['day'] > start_index_of_previous_year)]
    input_batch = priced_bitcoin[(priced_bitcoin['year'] >= START_YEAR) & (priced_bitcoin['year'] <= END_YEAR)]
    filtered_data = previous_year_batch.append(input_batch)
    filtered_data.insert(0, 'index', range(0, len(filtered_data)))
    filtered_data = filtered_data.reset_index(drop=True)
    return filtered_data

def preprocess_data(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    data_file_path = get_data_file_path()
    price_bitcoin_file_path = os.path.join(data_file_path, PRICED_BITCOIN_FILE_NAME)
    priced_bitcoin = pd.read_csv(price_bitcoin_file_path, sep=",")
    if (ALL_YEAR_INPUT_ALLOWED):
        pass
    else:
        priced_bitcoin = filter_data(priced_bitcoin)

# get normalized occurrence matrix in a flat format and merge with totaltx
    daily_occurrence_input = np.array([], dtype=np.float32)
    temp = np.array([], dtype=np.float32)
    
    for current_index, current_row in priced_bitcoin.iterrows():
        if(current_index<(window_size+prediction_horizon-1)):
            pass
        else:
            start_index = current_index-(window_size + prediction_horizon) + 1
            end_index = current_index - prediction_horizon
            temp = get_daily_occurrence_matrices(priced_bitcoin[start_index:end_index+1], current_row, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
        if daily_occurrence_input.size == 0:
            daily_occurrence_input = temp
        else:
            daily_occurrence_input = np.concatenate((daily_occurrence_input, temp), axis=0)

    return daily_occurrence_input

def print_results(predicted, test_target, test_days):
    myFile = open(LOG_FILE, 'a')
    for p, t, t_d in zip(predicted, test_target, test_days):
        myFile.write(str(p).strip("]").strip("[") + "\t" + str(t).strip("]").strip("[") + "\t" + str(t_d).strip("]").strip("[") + '\n')
    myFile.close()

def print_cost(total_cost):
    myFile = open(LOG_FILE, 'a')
    myFile.write('TOTAL_COST:' + str(total_cost) + '\n')
    myFile.close()

def run_print_model(train_input_list, train_target_list, test_input_list, test_target_list, train_list_days, test_list_days):
    total_cost = 0
    for index in range(0, len(train_input_list)):
        train_input = train_input_list[index]
        train_target = train_target_list[index]
        test_input = test_input_list[index]
        test_target = test_target_list[index]
        test_days = test_list_days[index]

        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
        train_target = scaler.fit_transform(np.asarray(train_target).reshape(-1,1))
        test_target = scaler.transform(np.asarray(test_target).reshape(-1,1))

        rf_regression = RandomForestRegressor(max_depth=2, random_state=0)
        rf_regression.fit(train_input, train_target)
        predicted = rf_regression.predict(test_input)

        predicted_ = scaler.inverse_transform(np.asarray(predicted).reshape(-1,1))
        price_ = scaler.inverse_transform(np.asarray(test_target).reshape(-1,1))

        cost = math.sqrt(mean_squared_error(price_, predicted_))
        total_cost = total_cost + cost
        print_results(predicted_, price_, test_days)
    print_cost(total_cost)
#----------------------------------------------------------------------------------------------------------------------#

def exclude_days(train_list, test_list):

    train_days_list = list()
    test_days_list = list()

    train_input_list = list()
    test_input_list = list()

    for index in range(0, len(train_list)):
        x_train = train_list[index]
        x_test = test_list[index]

        row, column = x_train.shape
        train_days = np.asarray(x_train[:, -1]).reshape(-1, 1)
        x_train = x_train[:, 0:column - 1]
        test_days = np.asarray(x_test[:, -1]).reshape(-1, 1)
        x_test = x_test[:, 0:column - 1]

        train_days_list.append(train_days)
        test_days_list.append(test_days)
        train_input_list.append(x_train)
        test_input_list.append(x_test)

    return train_input_list, test_input_list, train_days_list, test_days_list


def print_list(train_list, test_list):
    for index in range(0, len(train_list)):
        for training, test in zip(train_list[index][:,-1], test_list[index][:,-1]):
            print(str(training) + "\t" + str(test) + "\n")
    print("BITTI")

def train_test_split_(data):
    start_index = 0
    end_index = 0
    train_list = list()
    test_list = list()
    while((end_index+SLIDING_BATCH_SIZE) < data.shape[0]):
        end_index = end_index + SLIDING_BATCH_SIZE
        train_list.append(data[start_index:end_index])
        test_list.append(data[end_index:end_index+SLIDING_BATCH_SIZE])
        start_index = start_index + SLIDING_BATCH_SIZE
    return train_list, test_list

def split_input_target(x_train_list, x_test_list):

    train_input_list = list()
    train_target_list = list()
    test_input_list = list()
    test_target_list = list()

    for index in range(0, len(x_train_list)):
        x_train = x_train_list[index]
        x_test = x_test_list[index]

        row, column = x_train.shape
        train_target = np.asarray(x_train[:, -1]).reshape(-1, 1)
        train_input = x_train[:, 0:column - 1]
        test_target = np.asarray(x_test[:, -1]).reshape(-1, 1)
        test_input = x_test[:, 0:column - 1]

        train_input_list.append(train_input)
        train_target_list.append(train_target)
        test_input_list.append(test_input)
        test_target_list.append(test_target)

    return train_input_list, train_target_list, test_input_list, test_target_list

def initialize_setting(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    data = preprocess_data(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
    train_list, test_list = train_test_split_(data)
    x_train_list, x_test_list, train_list_days, test_list_days = exclude_days(train_list, test_list)
    train_input_list, train_target_list, test_input_list, test_target_list = split_input_target(x_train_list, x_test_list)

    return train_input_list, train_target_list, test_input_list, test_target_list, train_list_days, test_list_days

def print_initializer(window_size, prediction_horizon):
    myFile = open(LOG_FILE, 'a')
    myFile.write('IS_PRICE_OF_PREVIOUS_DAYS_ALLOWED:' + str(is_price_of_previous_days_allowed) + '\n')
    myFile.write('AGGREGATION_OF_PREVIOUS_DAYS_ALLOWED:' + str(aggregation_of_previous_days_allowed) + '\n')

    myFile.write('PREDICTION_HORIZON:' + str(prediction_horizon) + '\n')
    myFile.write('WINDOW_SIZE:' + str(window_size) + '\n')

parameter_dict = {0: dict({'is_price_of_previous_days_allowed':True, 'aggregation_of_previous_days_allowed':True}),
                  1: dict({'is_price_of_previous_days_allowed':True, 'aggregation_of_previous_days_allowed':False})}

for step in parameter_dict:
    gc.collect()
    evalParameter = parameter_dict.get(step)
    is_price_of_previous_days_allowed = evalParameter.get('is_price_of_previous_days_allowed')
    aggregation_of_previous_days_allowed = evalParameter.get('aggregation_of_previous_days_allowed')
    print("IS_PRICE_OF_PREVIOUS_DAYS_ALLOWED: ", is_price_of_previous_days_allowed)
    print("AGGREGATION_OF_PREVIOUS_DAYS_ALLOWED: ", aggregation_of_previous_days_allowed)
    for prediction_horizon in range(1, 8):
        print("PREDICTION_HORIZON: ", prediction_horizon)
        for window_size in range(1, 8):
            print('WINDOW_SIZE: ', window_size)
            print_initializer(window_size, prediction_horizon)
            train_input_list, train_target_list, test_input_list, test_target_list, train_list_days, test_list_days = initialize_setting(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
            run_print_model(train_input_list, train_target_list, test_input_list, test_target_list, train_list_days, test_list_days)











