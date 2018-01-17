import pandas as pd
from sklearn import preprocessing
import numpy as np, tensorflow as tf
import os
import math
from sklearn.metrics import mean_squared_error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


ALL_YEAR_INPUT_ALLOWED = False
START_YEAR = 2016
END_YEAR = 2017
NUMBER_OF_CLASSES = 1

SLIDING_BATCH_SIZE = 60
CHAINLET_ALLOWED = True

METHOD = "amo"
PRICED_BITCOIN_FILE_NAME = "D:\\Bitcoin\\pricedBitcoin2009-2018.csv"
DAILY_OCCURRENCE_FILE_PATH = "D:\\Bitcoin\\createddata\\daily" + METHOD + "matrices\\"
RESULT_FOLDER = "D:\\Bitcoin\\createddata\\results\\"
RESULT_FILE = METHOD + 'slidingPrediction.csv'

if os.path.isfile(RESULT_FOLDER + RESULT_FILE):
    os.remove(RESULT_FOLDER + RESULT_FILE)
ROW = -1
COLUMN = -1
TEST_SPLIT = 0.2

#DEEP_LEARNING_PARAMETERS
REGULARIZATION_FOR_LOG = 0.0000001
LEARNING_RATE = 0.01
STEP_NUMBER = 20000
UNITS_OF_HIDDEN_LAYER_1 = 128
UNITS_OF_HIDDEN_LAYER_2 = 64
EPSILON = 1e-3
DISPLAY_STEP = int(STEP_NUMBER / 10)
LOG_RETURN_USED = False



def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, EPSILON)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, EPSILON)

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
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['year']).reshape(1,1)), axis=1)
    return occurrence_input

def get_normalized_matrix_from_file(day, year, totaltx):
    daily_occurrence_file_path = os.path.join(DAILY_OCCURRENCE_FILE_PATH,
                                              METHOD + str(year) + '{:03}'.format(day) + ".csv")
    daily_occurrence_matrix = pd.read_csv(daily_occurrence_file_path, sep=",", header=None).values
    return np.asarray(daily_occurrence_matrix).reshape(1, daily_occurrence_matrix.size)/totaltx


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
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_NAME, sep=",")

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


def print_results(predictedPrice, realPrice, test_years, test_days):
    myFile = open(RESULT_FOLDER + RESULT_FILE, 'a')
    prefix = str(is_price_of_previous_days_allowed) + "\t" + str(aggregation_of_previous_days_allowed) + '\t' + str(
        window_size) + '\t' + str(prediction_horizon) + '\t'

    for pred, real, year, day in zip(predictedPrice, realPrice, test_years, test_days):
        myFile.write(prefix + "\t" +
                     str(SLIDING_BATCH_SIZE) + "\t" +
                     str(pred[0]) + "\t" +
                     str(real[0]) + "\t" +
                     str(year[0]) + "\t" +
                     str(int(day[0])) + '\n')
    myFile.close()

def print_cost(total_cost):
    myFile = open(RESULT_FOLDER + RESULT_FILE, 'a')
    myFile.write('TOTAL_COST:' + str(total_cost) + '\n')
    myFile.close()

def run_print_model(input_number, train_input_list, train_target_list, test_input_list, test_target_list, train_year_list, test_year_list, train_list_days, test_list_days):
    total_cost = 0
    for index in range(0, len(train_input_list)):
        train_input = train_input_list[index]
        train_target = train_target_list[index]
        test_input = test_input_list[index]
        test_target = test_target_list[index]
        test_days = test_list_days[index]
        test_years = test_year_list[index]

        if(not LOG_RETURN_USED):
            scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
            train_target = scaler.fit_transform(np.asarray(train_target).reshape(-1,1))
            test_target = scaler.transform(np.asarray(test_target).reshape(-1,1))

        predicted_price = []
        with tf.Graph().as_default(), tf.Session() as sess, tf.device('/cpu:0'):
            #--------------------------------------------------------------------------------------------------------------#
            TRAIN_NUMBER = train_input.shape[0]

            #initial_weights
            w1_initial = np.random.normal(size=(input_number, UNITS_OF_HIDDEN_LAYER_1)).astype(np.float32)
            w2_initial = np.random.normal(size=(UNITS_OF_HIDDEN_LAYER_1, UNITS_OF_HIDDEN_LAYER_2)).astype(np.float32)
            w3_initial = np.random.normal(size=(UNITS_OF_HIDDEN_LAYER_2, NUMBER_OF_CLASSES)).astype(np.float32)

            # Placeholders
            input = tf.placeholder(tf.float32, shape=[None, input_number])
            price = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])

            # Layer 1
            w1 = tf.Variable(w1_initial)
            z1 = tf.matmul(input, w1)
            bn1 = batch_norm_wrapper(z1, True)
            l1 = tf.nn.relu(bn1)

            #Layer 2
            w2 = tf.Variable(w2_initial)
            z2 = tf.matmul(l1, w2)
            bn2 = batch_norm_wrapper(z2, True)
            l2 = tf.nn.relu(bn2)

            # Softmax
            w3 = tf.Variable(w3_initial)
            b3 = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))
            predicted = tf.nn.relu(tf.matmul(l2, w3) + b3)

            # Loss, Optimizer and Predictions
            if(LOG_RETURN_USED):
                rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(tf.log(price), predicted)))
            else:
                rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(price, predicted)))
            train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(rmse)
            #--------------------------------------------------------------------------------------------------------------#
            for v in tf.trainable_variables():
                sess.run(tf.variables_initializer([v]))
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(STEP_NUMBER):
                sample = np.random.randint(TRAIN_NUMBER, size=5)
                batch_xs = train_input[sample][:]
                batch_ys = train_target[sample][:]
                sess.run(train_step, feed_dict={input: batch_xs, price: batch_ys})
                if i == STEP_NUMBER-1:
                    predicted_price.append(sess.run([predicted], feed_dict={input: test_input, price: test_target})[0])

            if(not LOG_RETURN_USED):
                predicted_ = scaler.inverse_transform(predicted_price[0])
                price_ = scaler.inverse_transform(test_target)
            else:
                predicted_ = predicted_price[0]
                price_ = np.log(test_target)

            total_cost = total_cost + math.sqrt(mean_squared_error(price_, predicted_))
            print_results(predicted_, price_, test_years, test_days)
            # print_cost(total_cost)
#----------------------------------------------------------------------------------------------------------------------#

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
        train_days = np.asarray(x_train[:, column]).reshape(-1, 1)
        test_days = np.asarray(x_test[:, column]).reshape(-1, 1)

        train_year = np.asarray(x_train[:, column-1]).reshape(-1, 1)
        test_year = np.asarray(x_test[:, column-1]).reshape(-1, 1)

        x_train = x_train[:, 0:column-2]
        x_test = x_test[:, 0:column-2]

        train_days_list.append(train_days)
        test_days_list.append(test_days)
        train_year_list.append(train_year)
        test_year_list.append(test_year)
        train_input_list.append(x_train)
        test_input_list.append(x_test)

    return train_input_list, test_input_list, train_year_list, test_year_list, train_days_list, test_days_list


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

    return column-1, train_input_list, train_target_list, test_input_list, test_target_list

def initialize_setting(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    data = preprocess_data(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
    train_list, test_list = train_test_split_(data)
    x_train_list, x_test_list, train_year_list, test_year_list, train_list_days, test_list_days = exclude_days(train_list, test_list)
    input_number, train_input_list, train_target_list, test_input_list, test_target_list = split_input_target(x_train_list, x_test_list)

    return input_number, train_input_list, train_target_list, test_input_list, test_target_list, train_year_list, test_year_list,  train_list_days, test_list_days


parameter_dict = {0: dict({'is_price_of_previous_days_allowed':True, 'aggregation_of_previous_days_allowed':True}),
                  1: dict({'is_price_of_previous_days_allowed':True, 'aggregation_of_previous_days_allowed':False})}



for step in parameter_dict:
    evalParameter = parameter_dict.get(step)
    is_price_of_previous_days_allowed = evalParameter.get('is_price_of_previous_days_allowed')
    aggregation_of_previous_days_allowed = evalParameter.get('aggregation_of_previous_days_allowed')
    for prediction_horizon in range(1, 8):
        for window_size in range(1, 8):
            print('window: ', window_size,
                  "horizon:", prediction_horizon,
                  "batch_size:", SLIDING_BATCH_SIZE,
                  "priced:", is_price_of_previous_days_allowed,
                  "Aggregated:", aggregation_of_previous_days_allowed)
            input_number, train_input_list, train_target_list, test_input_list, test_target_list, train_year_list, test_year_list, train_list_days, test_list_days = initialize_setting(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
            run_print_model(input_number, train_input_list, train_target_list, test_input_list, test_target_list, train_year_list, test_year_list, train_list_days, test_list_days)











