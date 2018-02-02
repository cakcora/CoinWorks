import pandas as pd
from sklearn import preprocessing
import numpy as np, tensorflow as tf
import os
import math
from sklearn.metrics import mean_squared_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


ALL_YEAR_INPUT_ALLOWED = False
START_YEAR = 2017
END_YEAR = 2017
NUMBER_OF_CLASSES = 1

METHOD = ""
PRICED_BITCOIN_FILE_NAME = "D:\\Bitcoin\\pricedBitcoin2009-2018.csv"
BETTI_NUMBER_FILE_PATH = "D:\\Bitcoin\\createddata\\betti_numbers\\"
RESULT_FOLDER = "D:\\Bitcoin\\createddata\\results\\"
RESULT_FILE = METHOD + 'betti_prediction.csv'


if os.path.isfile(RESULT_FOLDER + RESULT_FILE):
    os.remove(RESULT_FOLDER + RESULT_FILE)


# DEEP_LEARNING_PARAMETERS
LEARNING_RATE = 0.005
STEP_NUMBER = 2000
UNITS_OF_HIDDEN_LAYER_1 = 512
UNITS_OF_HIDDEN_LAYER_2 = 256
UNITS_OF_HIDDEN_LAYER_3 = 128
UNITS_OF_HIDDEN_LAYER_4 = 64
DISPLAY_STEP = int(STEP_NUMBER / 10)
FILTER = 100


def merge_data(previous_daily_data, betti_number_0, betti_number_1):
    if previous_daily_data.size==0:
        previous_daily_data = np.concatenate((betti_number_0, betti_number_1), axis=1)
    else:
        merged_betti_numbers = np.concatenate((betti_number_0, betti_number_1), axis=1)
        previous_daily_data = np.concatenate((previous_daily_data, merged_betti_numbers), axis=1)
    return previous_daily_data

def get_daily_matrices(priced_bitcoin, current_row, priced, betti_allowed, log_return):
    previous_price_data = np.array([], dtype=np.float32)
    previous_daily_data = np.array([], dtype=np.float32)
    merged_data = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])
            betti_number_0 = get_betti_numbers(row['day'], os.path.join(BETTI_NUMBER_FILE_PATH, "betti_" + str(0) + "(" + str(FILTER) + ").csv"))
            betti_number_1 = get_betti_numbers(row['day'], os.path.join(BETTI_NUMBER_FILE_PATH, "betti_" + str(1) + "(" + str(FILTER) + ").csv"))
            previous_daily_data = merge_data(previous_daily_data, betti_number_0, betti_number_1)
    if betti_allowed:
        merged_data = previous_daily_data
    if priced:
        if merged_data.size == 0:
            merged_data = np.asarray(previous_price_data).reshape(1, -1)
        else:
            merged_data = np.concatenate((previous_daily_data, np.asarray(previous_price_data).reshape(1, -1)), axis=1)

    if log_return:
        merged_data = np.concatenate((merged_data, np.asarray(current_row['log_return']).reshape(1, 1)), axis=1)
    else:
        merged_data = np.concatenate((merged_data, np.asarray(current_row['price']).reshape(1, 1)), axis=1)

    merged_data = np.concatenate((merged_data, np.asarray(current_row['day']).reshape(1, 1)), axis=1)
    merged_data = np.concatenate((merged_data, np.asarray(current_row['year']).reshape(1, 1)), axis=1)
    return merged_data

def get_betti_numbers(row_day, betti_number_file_name):
    betti_numbers = pd.read_csv(betti_number_file_name, sep=",",)
    row_bettis = betti_numbers[betti_numbers['day'] == row_day]
    row_bettis = row_bettis.drop(['day'], axis=1)

    return np.asarray(row_bettis).reshape(1, row_bettis.size)

def filter_data(priced_bitcoin, train_slide_length):

    end_day_of_previous_year = max(priced_bitcoin[priced_bitcoin['year'] == START_YEAR - 1]["day"].values)
    start_index_of_previous_year = end_day_of_previous_year - train_slide_length
    previous_year_batch = priced_bitcoin[
        (priced_bitcoin['year'] == START_YEAR - 1) & (priced_bitcoin['day'] > start_index_of_previous_year)]
    input_batch = priced_bitcoin[(priced_bitcoin['year'] >= START_YEAR) & (priced_bitcoin['year'] <= END_YEAR)]
    filtered_data = previous_year_batch.append(input_batch)
    filtered_data.insert(0, 'index', range(0, len(filtered_data)))
    filtered_data = filtered_data.reset_index(drop=True)
    return filtered_data


def scale_prices(priced_bitcoin, log_return):

    price = priced_bitcoin['price'].values
    log_return_list = list()
    if log_return:
        for index in range(len(price)):
            if index == 0:
                log_return_list.append(float(0))
            else:
                log_return_list.append(np.log(price[index]) - np.log(price[index - 1]))
        log_return_list = pd.DataFrame(log_return_list, columns=["log_return"])
        log_return_list.reset_index(drop=True, inplace=True)
        priced_bitcoin = pd.concat([priced_bitcoin, log_return_list], axis=1)

    return priced_bitcoin


def preprocess_data(window, horizon, priced, betti_allowed, log_return, train_slide_length):
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_NAME, sep=",")

    if (ALL_YEAR_INPUT_ALLOWED):
        pass
    else:
        priced_bitcoin = filter_data(priced_bitcoin, train_slide_length)

    priced_bitcoin = scale_prices(priced_bitcoin, log_return)
    daily_input = np.array([], dtype=np.float32)
    temp = np.array([], dtype=np.float32)

    for current_index, current_row in priced_bitcoin.iterrows():
        if (current_index < (window + horizon - 1)):
            pass
        else:
            start_index = current_index - (window + horizon) + 1
            end_index = current_index - horizon
            temp = get_daily_matrices(priced_bitcoin[start_index:end_index + 1], current_row, priced, betti_allowed, log_return)
        if daily_input.size == 0:
            daily_input = temp
        else:
            daily_input = np.concatenate((daily_input, temp), axis=0)

    total_column = daily_input.shape[1]
    matrix_column = 3 # target, day, year

    scaler = preprocessing.MinMaxScaler()
    daily_input[:,0:total_column-matrix_column] = scaler.fit_transform(daily_input[:,0:total_column-matrix_column])

    target_scaler = preprocessing.MinMaxScaler()

    if(log_return):
        pass
    else:
        daily_input[:,-3] = np.asarray(target_scaler.fit_transform(np.asarray(daily_input[:,-3]).reshape(-1,1))).reshape(np.asarray(daily_input[:,-3]).shape)

    return target_scaler, daily_input


def print_results(predicted_price, real_price, test_years, test_days):
    myFile = open(RESULT_FOLDER + RESULT_FILE, 'a')
    prefix = str(priced) + "\t" + str(betti_allowed) + '\t' + str(
        window) + '\t' + str(horizon)

    for pred, real, year, day in zip(predicted_price, real_price, test_years, test_days):
        myFile.write(prefix + "\t" +
                     str(train_slide_length) + "\t" +
                     str(test_slide_length) + "\t" +
                     str(pred[0]) + "\t" +
                     str(real[0]) + "\t" +
                     str(int(year[0])) + "\t" +
                     str(int(day[0])) + '\n')
    myFile.close()


def print_cost(total_cost):
    myFile = open(RESULT_FOLDER + RESULT_FILE, 'a')
    myFile.write('TOTAL_COST:' + str(total_cost) + '\n')
    myFile.close()


def build_graph(input_number):

    # Placeholders
    input = tf.placeholder(tf.float32, shape=[None, input_number])
    price = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])

    # Layer 1
    w1 = tf.Variable(tf.random_uniform([input_number, UNITS_OF_HIDDEN_LAYER_1], minval=-math.sqrt(6/(input_number+UNITS_OF_HIDDEN_LAYER_1)), maxval=math.sqrt(6/(input_number+UNITS_OF_HIDDEN_LAYER_1))))
    z1 = tf.matmul(input, w1)
    l1 = tf.nn.tanh(z1)
    l1_dropout = tf.nn.dropout(l1, 0.8)

    # Layer 2
    w2 = tf.Variable(tf.random_uniform([UNITS_OF_HIDDEN_LAYER_1, UNITS_OF_HIDDEN_LAYER_2], minval=-math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_1+UNITS_OF_HIDDEN_LAYER_2)), maxval=math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_1+UNITS_OF_HIDDEN_LAYER_2))))
    z2 = tf.matmul(l1_dropout, w2)
    l2 = tf.nn.tanh(z2)
    l2_dropout = tf.nn.dropout(l2, 0.8)

    # Layer 3
    w3 = tf.Variable(tf.random_uniform([UNITS_OF_HIDDEN_LAYER_2, UNITS_OF_HIDDEN_LAYER_3], minval=-math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_2+UNITS_OF_HIDDEN_LAYER_3)), maxval=math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_2+UNITS_OF_HIDDEN_LAYER_3))))
    z3 = tf.matmul(l2_dropout, w3)
    l3 = tf.nn.tanh(z3)
    l3_dropout = tf.nn.dropout(l3, 0.8)

    # Layer 4
    w4 = tf.Variable(tf.random_uniform([UNITS_OF_HIDDEN_LAYER_3, UNITS_OF_HIDDEN_LAYER_4], minval=-math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_3+UNITS_OF_HIDDEN_LAYER_4)), maxval=math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_3+UNITS_OF_HIDDEN_LAYER_4))))
    z4 = tf.matmul(l3_dropout, w4)
    l4 = tf.nn.tanh(z4)
    l4_dropout = tf.nn.dropout(l4, 0.8)

    # Linear
    w5 = tf.Variable(tf.random_uniform([UNITS_OF_HIDDEN_LAYER_4, NUMBER_OF_CLASSES], minval=-math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_4+NUMBER_OF_CLASSES)), maxval=math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_4+NUMBER_OF_CLASSES))))
    b5 = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))
    predicted = tf.matmul(l4_dropout, w5) + b5

    rmse = tf.losses.mean_squared_error(price, predicted)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(rmse)

    return (input, price), train_step, predicted, rmse, tf.train.Saver()


def run_print_model(scaler, input_number, log_return, train_input_list, train_target_list, test_input_list, test_target_list,
                    test_year_list, test_list_days):
    total_cost = -1
    for index in range(0, len(train_input_list)):
        train_input = train_input_list[index]
        train_target = train_target_list[index]
        test_input = test_input_list[index]
        test_target = test_target_list[index]
        test_days = test_list_days[index]
        test_years = test_year_list[index]


        tf.reset_default_graph()
        (input, price), train_step, predicted, loss, saver = build_graph(input_number)

        train_loss_list = []
        test_loss_list = []
        predicted_list = []
# ---------------------------------------------------------------------------------------------------------------------#
        TRAIN_NUMBER = train_input.shape[0]
        if(TRAIN_NUMBER == 0):
            print("TOTAL TRAIN IS ZERO, MODEL CANNOT BE GENERATED")
        else:
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                for i in range(STEP_NUMBER):
                    sample = np.random.randint(TRAIN_NUMBER, size=1)
                    batch_xs = train_input[sample][:]
                    batch_ys = train_target[sample][:]
                    train_step.run(session=sess, feed_dict={input: batch_xs, price: batch_ys})
                    if i % DISPLAY_STEP == 0:
                        train_loss_list.append(sess.run([loss], feed_dict={input: train_input, price: train_target})[0])
                        sess.run([predicted], feed_dict={input: train_input, price: train_target})[0]
                    if i == STEP_NUMBER - 1:
                        test_loss_list.append(sess.run([loss], feed_dict={input: test_input, price: test_target})[0])
                        predicted_list.append(sess.run([predicted], feed_dict={input: test_input, price: test_target})[0])
                if log_return:
                    predicted_ = predicted_list[0]
                    price_ = test_target
                else:
                    predicted_ = scaler.inverse_transform(predicted_list[0])
                    price_ = scaler.inverse_transform(test_target)

                print_results(predicted_, price_, test_years, test_days)
# ---------------------------------------------------------------------------------------------------------------------#
def exclude_days(train_list, test_list):

    train_days_list = list()
    test_days_list = list()

    train_year_list = list()
    test_year_list = list()

    train_input_list = list()
    test_input_list = list()

    for index in range(0, len(train_list)):
        x_train = train_list[index]
        x_test = test_list[index]

        row, column = x_train.shape
        train_year = np.asarray(x_train[:, -1]).reshape(-1, 1)
        test_year = np.asarray(x_test[:, -1]).reshape(-1, 1)

        train_days = np.asarray(x_train[:, -2]).reshape(-1, 1)
        test_days = np.asarray(x_test[:, -2]).reshape(-1, 1)

        x_train = x_train[:, 0:column - 2]
        x_test = x_test[:, 0:column - 2]

        train_days_list.append(train_days)
        test_days_list.append(test_days)
        train_year_list.append(train_year)
        test_year_list.append(test_year)
        train_input_list.append(x_train)
        test_input_list.append(x_test)

    return train_input_list, test_input_list, train_year_list, test_year_list, train_days_list, test_days_list


def print_list(train_list, test_list):
    for index in range(0, len(train_list)):
        for training, test in zip(train_list[index][:, -1], test_list[index][:, -1]):
            print(str(training) + "\t" + str(test) + "\n")
    print("BITTI")


def train_test_split_(data, train_slide_length, test_slide_length, window):

    start_index = 0
    end_index = 0
    train_list = list()
    test_list = list()
    while ((end_index + test_slide_length) < data.shape[0]):
        end_index = start_index + train_slide_length - window
        train_list.append(data[start_index:end_index])
        test_list.append(data[end_index:end_index + test_slide_length])
        start_index = start_index + test_slide_length
    return train_list, test_list


def split_input_target(x_train_list, x_test_list):

    train_input_list = list()
    train_target_list = list()
    test_input_list = list()
    test_target_list = list()
    column = -1
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

    return column - 1, train_input_list, train_target_list, test_input_list, test_target_list


def initialize_setting(window, horizon, priced, betti_allowed, log_return, train_slide_length, test_slide_length):

    scaler, data = preprocess_data(window, horizon, priced, betti_allowed, log_return, train_slide_length)
    train_list, test_list = train_test_split_(data, train_slide_length, test_slide_length, window)
    x_train_list, x_test_list, train_year_list, test_year_list, train_list_days, test_list_days = exclude_days(
        train_list, test_list)
    input_number, train_input_list, train_target_list, test_input_list, test_target_list = split_input_target(
        x_train_list, x_test_list)

    return scaler, input_number, train_input_list, train_target_list, test_input_list, test_target_list, train_year_list, test_year_list, train_list_days, test_list_days

parameter_dict = {0: dict({'priced': True, 'betti_allowed': True, 'log_return': True, "train_slide_length" : 7, "test_slide_length":1})}

for step in parameter_dict:
    evalParameter = parameter_dict.get(step)
    priced = evalParameter.get('priced')
    betti_allowed = evalParameter.get('betti_allowed')
    log_return = evalParameter.get('log_return')
    train_slide_length = evalParameter.get('train_slide_length')
    test_slide_length = evalParameter.get('test_slide_length')
    if betti_allowed == False and priced == False:
        print("!!!!Input can not be empty!!!!")
        break
    if train_slide_length<=0:
        print("Train slide length can not be negative or zero")
        break
    if train_slide_length>365:
        print("Training slide length can not be bigger than 365")
        break
    if test_slide_length<=0:
        print("Test slide length can not be negative or zero")
        break
    if test_slide_length>365:
        print("Test slide length can not be bigger than 365")
        break
    for horizon in range(1,10):
        for window in range(1,10):
            if train_slide_length >= (horizon + window):
                print('window: ', window,
                    "horizon:", horizon,
                    "train_slide_length:", train_slide_length,
                    "test_slide_length:", test_slide_length,
                    "priced:", priced,
                    "betti_allowed:", betti_allowed,
                    "log_return:", log_return)
                scaler, input_number, train_input_list, train_target_list, test_input_list, test_target_list, train_year_list, test_year_list, train_list_days, test_list_days = initialize_setting(
                    window, horizon, priced, betti_allowed, log_return, train_slide_length, test_slide_length)
                run_print_model(scaler, input_number, log_return, train_input_list, train_target_list, test_input_list, test_target_list, test_year_list, test_list_days)
            else:
                print("Sum of horizon and window is bigger than train slide length")
                print('window: ', window,
                      "horizon:", horizon,
                      "train_slide_length:", train_slide_length)
print("TASK COMPLETED")
