# LSTM for international airline passengers problem with regression framing
import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.layers.core import Dense, Dropout


PRICED_BITCOIN_FILE_PATH = "C:\\Users\\nca150130\\Desktop\\matrix\\pricedBitcoin.csv"
DAILY_OCCURRENCE_FILE_PATH = "C:\\Users\\nca150130\\Desktop\\matrix\\dailyOccmatrices\\"

NUMBER_OF_CLASSES = 1

ROW = -1
COLUMN = -1
TEST_SPLIT = 0.8

#DEEP_LEARNING_PARAMETERS
LEARNING_RATE = 0.01
BATCH_SIZE = 2
STEP_NUMBER = 5000
UNITS_OF_HIDDEN_LAYER_1 = 8
UNITS_OF_HIDDEN_LAYER_2 = 8
DISPLAY_STEP = 10
ALL_YEAR_INPUT_ALLOWED = False
YEAR = 2017
DROP_OUT_PERCENTAGE = 0.2

def load_data():
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_PATH, sep=",")
    if (ALL_YEAR_INPUT_ALLOWED):
        pass
    else:
        priced_bitcoin = priced_bitcoin[priced_bitcoin['year'] == YEAR].reset_index(drop=True)
    return priced_bitcoin

def exclude_days(train, test):
    row, column = train.shape
    train_days = np.asarray(train[:, -1]).reshape(-1, 1)
    x_train = train[:, 0:column - 1]
    test_days = np.asarray(test[:, -1]).reshape(-1, 1)
    x_test = test[:, 0:column - 1]
    return x_train, x_test, train_days, test_days

def preprocess_data(data, window_size, prediction_horizon):
    processed_data = np.array([], dtype=np.float32)
    for current_index, current_row in data.iterrows():
        if (current_index < (window_size + prediction_horizon - 1)):
            pass
        else:
            start_index = current_index - (window_size + prediction_horizon) + 1
            end_index = current_index - prediction_horizon
            temp = np.concatenate((data[start_index:end_index + 1]['price'].reshape(1,-1), np.asarray(current_row['price']).reshape(1,1)), axis=1)
            temp = np.concatenate((temp, np.asarray(current_row['day']).reshape(1,1)), axis=1)
            if processed_data.size == 0:
                processed_data = temp
            else:
                processed_data = np.concatenate((processed_data, temp), axis=0)

    return processed_data

def normalize_data(data):
    price = np.asarray(data['price'].values).reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    input = scaler.fit_transform(price)
    data['price'] = input

    return scaler, data

def train_test_split_(data):
    #np.random.shuffle(data)
    train_size = int(len(data)*TEST_SPLIT)
    remaining = train_size%BATCH_SIZE
    if remaining != 0:
        train_size = train_size - remaining
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    return train, test

def initialize_setting(window_size, prediction_horizon):
    data = load_data()
    scaler, data = normalize_data(data)
    data = preprocess_data(data, window_size, prediction_horizon)
    train, test = train_test_split_(data)
    x_train, x_test, train_days, test_days = exclude_days(train, test)
    row, column = x_train.shape
    train_target = np.asarray(x_train[:, -1]).reshape(-1, 1)
    train_input = x_train[:, 0:column - 1]
    test_target = np.asarray(x_test[:, -1]).reshape(-1, 1)
    test_input = x_test[:, 0:column - 1]
    return scaler, train_input, train_target, test_input, test_target, train_days, test_days

def re_define_model(model, window_size, batch_size):
    new_model = get_nn(window_size, batch_size)
    # copy weights
    old_weights = model.get_weights()
    new_model.set_weights(old_weights)
    # compile model
    new_model.compile(loss='mean_squared_error', optimizer='adam')
    return new_model

def build_rnn(train_input, train_target, window_size, batch_size):
    model = get_nn(window_size, batch_size)
    model.compile(loss="mean_squared_error", optimizer='adam')
    for i in range(STEP_NUMBER):
        model.fit(train_input, train_target, epochs=1, batch_size=batch_size, verbose=2)
        model.reset_states()
    new_model = re_define_model(model, window_size, 1)
    return new_model

def get_nn(window_size, batch_size):
    model = Sequential()
    model.add(LSTM(UNITS_OF_HIDDEN_LAYER_1, batch_input_shape=(batch_size, 1, window_size), stateful=True, return_sequences=True))
    model.add(Dropout(DROP_OUT_PERCENTAGE))
    model.add(LSTM(UNITS_OF_HIDDEN_LAYER_2, batch_input_shape=(batch_size, 1, window_size), stateful=True))
    model.add(Dropout(DROP_OUT_PERCENTAGE))
    model.add(Dense(1, activation='sigmoid'))
    return model

def get_model_prediction(rnn, test_X, batch_size):
    predicted_list = list()
    for i in range(len(test_X)):
        x = test_X[i]
        x = x.reshape(batch_size, 1, window_size)
        predicted = rnn.predict(x, batch_size=batch_size)
        predicted_list.append(predicted)
    return np.asarray(predicted_list).reshape(-1,1)

def print_model(scaler, rnn, test_X, test_Y, test_days, batch_size, window_size, prediction_horizon):
    myFile = open('C:\\Users\\nca150130\\PycharmProjects\\CoinWorks\\src\\main\\python\\results\\rnn_order___' + str(YEAR) + ".csv", 'a')
    predicted = get_model_prediction(rnn, test_X, batch_size)
    test_predicted = scaler.inverse_transform(predicted)
    price = scaler.inverse_transform(test_Y)
    previous_price = np.asarray(scaler.inverse_transform(test_X.reshape(-1,window_size)[:,-1].reshape(-1,1)))

    myFile.write("WINDOW_SIZE: " + str(window_size) + "\n")
    myFile.write("PREDICTION_HORIZON: " + str(prediction_horizon) + "\n")

    original_log_return = np.log(np.asarray(price).reshape(-1,1)/previous_price)
    predicted_log_return = np.log(np.asarray(test_predicted).reshape(-1,1)/previous_price)

    for i in range(test_days.shape[0]):
        predicted_, price_, test_day, original_log_return_, predicted_log_return_ = test_predicted[i], price[i], test_days[i], original_log_return[i], predicted_log_return[i]
        result = str(predicted_) + "," + str(price_) + "," + str(test_day) + "," + str(original_log_return_) + "," + str(predicted_log_return_) + "\n"
        myFile.write(result.replace("[", "").replace("]", ""))
    myFile.close()


window_size = 4
prediction_horizon = 2
scaler, train_input, train_target, test_input, test_target, train_days, test_days = initialize_setting(window_size, prediction_horizon)

train_X = np.reshape(train_input, (train_input.shape[0], 1, train_input.shape[1]))
train_Y = np.reshape(train_target,(-1, 1))
test_X = np.reshape(test_input, (test_input.shape[0], 1, test_input.shape[1]))
test_Y = np.reshape(test_target,(-1, 1))

rnn = build_rnn(train_X, train_Y, window_size, BATCH_SIZE)
print_model(scaler, rnn, test_X, test_Y, test_days, 1, window_size, prediction_horizon)


