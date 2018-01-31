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


CHAINLET_ALLOWED = True

METHOD = ""
PRICED_BITCOIN_FILE_NAME = "D:\\Bitcoin\\pricedBitcoin2009-2018.csv"
BETTI_NUMBER_FILE_PATH = "D:\\Bitcoin\\createddata\\betti_numbers\\"
RESULT_FOLDER = "D:\\Bitcoin\\createddata\\results\\"
RESULT_FILE = METHOD + 'betti_prediction.csv'

if os.path.isfile(RESULT_FOLDER + RESULT_FILE):
    os.remove(RESULT_FOLDER + RESULT_FILE)

ROW = -1
COLUMN = -1
TEST_SPLIT = 0.2

#DEEP_LEARNING_PARAMETERS
REGULARIZATION_FOR_LOG = 0.0000001
LEARNING_RATE = 0.01
STEP_NUMBER = 20000
UNITS_OF_HIDDEN_LAYER_1 = 256
UNITS_OF_HIDDEN_LAYER_2 = 128
UNITS_OF_HIDDEN_LAYER_3 = 64
EPSILON = 1e-3
DISPLAY_STEP = int(STEP_NUMBER/10)
LOG_RETURN_USED = True
FILTER = 100

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

def merge_data(occurrence_data, betti_number_0, betti_number_1):
    if occurrence_data.size==0:
        occurrence_data = np.concatenate((betti_number_0, betti_number_1), axis=1)
    else:
        merged_betti_numbers = np.concatenate((betti_number_0, betti_number_1), axis=1)
        occurrence_data = np.concatenate((occurrence_data, merged_betti_numbers), axis=1)
    return occurrence_data

def get_daily_betti_numbers(priced_bitcoin, current_row, previous_day_price, is_price_of_previous_days_allowed):
    previous_price_data = np.array([], dtype=np.float32)
    occurrence_data = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])
            betti_number_0 = get_betti_numbers(row['day'], os.path.join(BETTI_NUMBER_FILE_PATH, "betti_" + str(0) + "(" + str(FILTER) + ").csv"))
            betti_number_1 = get_betti_numbers(row['day'], os.path.join(BETTI_NUMBER_FILE_PATH, "betti_" + str(1) + "(" + str(FILTER) + ").csv"))
            occurrence_data = merge_data(occurrence_data, betti_number_0, betti_number_1)
    if(is_price_of_previous_days_allowed):
        if(CHAINLET_ALLOWED):
            occurrence_data = np.concatenate((occurrence_data, np.asarray(previous_price_data).reshape(1,-1)), axis=1)
        else:
            occurrence_data = np.asarray(previous_price_data).reshape(1,-1)
    if(LOG_RETURN_USED):
        log_return = np.log(current_row['price']) - np.log(float(previous_day_price))
        occurrence_input = np.concatenate((occurrence_data, np.asarray(log_return).reshape(1,1)), axis=1)
    else:
        occurrence_input = np.concatenate((occurrence_data, np.asarray(current_row['price']).reshape(1,1)), axis=1)
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['day']).reshape(1,1)), axis=1)
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['year']).reshape(1,1)), axis=1)
    return occurrence_input

def get_betti_numbers(row_day, betti_number_file_name):
    betti_numbers = pd.read_csv(betti_number_file_name, sep=",",)
    row_bettis = betti_numbers[betti_numbers['day'] == row_day]
    row_bettis = row_bettis.drop(['day'], axis=1)
    normalizer = preprocessing.Normalizer()
    normalized_betti_numbers = normalizer.transform(row_bettis)
    return np.asarray(normalized_betti_numbers).reshape(1, normalized_betti_numbers.size)

def preprocess_data(window_size, prediction_horizon, is_price_of_previous_days_allowed, slide_length_for_train):
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_NAME, sep=",")

    if (ALL_YEAR_INPUT_ALLOWED):
        pass
    else:
        priced_bitcoin = priced_bitcoin[(priced_bitcoin['year'] == START_YEAR)]
        priced_bitcoin.insert(0, 'index', range(0, len(priced_bitcoin)))
        priced_bitcoin = priced_bitcoin.reset_index(drop=True)

    daily_betties_input = np.array([], dtype=np.float32)
    temp = np.array([], dtype=np.float32)

    for current_index, current_row in priced_bitcoin.iterrows():
        if(current_index<(window_size+prediction_horizon-1)):
            pass
        else:
            start_index = current_index-(window_size + prediction_horizon) + 1
            end_index = current_index - prediction_horizon
            previous_day_price= priced_bitcoin.loc[current_index-1, ['price']].values
            temp = get_daily_betti_numbers(priced_bitcoin[start_index:end_index+1], current_row, previous_day_price, is_price_of_previous_days_allowed)
        if daily_betties_input.size == 0:
            daily_betties_input = temp
        else:
            daily_betties_input = np.concatenate((daily_betties_input, temp), axis=0)
    return daily_betties_input

def print_results(predictedPrice, realPrice, test_years, test_days):
    myFile = open(RESULT_FOLDER + RESULT_FILE, 'a')
    prefix = str(is_price_of_previous_days_allowed) + "\t" + str(window) + '\t' + str(horizon)

    for pred, real, year, day in zip(predictedPrice, realPrice, test_years, test_days):
        myFile.write(prefix + "\t" +
                     str(slide_length_for_train) + "\t" +
                     str(slide_length_for_test) + "\t" +
                     str(pred[0]) + "\t" +
                     str(real[0]) + "\t" +
                     str(int(year[0])) + "\t" +
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

        if(LOG_RETURN_USED):
            pass
        else:
            scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
            train_target = scaler.fit_transform(np.asarray(train_target).reshape(-1,1))
            test_target = scaler.transform(np.asarray(test_target).reshape(-1,1))

        predicted_price = []
        with tf.Graph().as_default(), tf.Session() as sess, tf.device('/cpu:0'):
            #--------------------------------------------------------------------------------------------------------------#
            TRAIN_NUMBER = train_input.shape[0]

            #initial_weights
            w1_initial = np.random.normal(size=(input_number, UNITS_OF_HIDDEN_LAYER_1)).astype(np.float32)*math.sqrt(2/input_number)
            w2_initial = np.random.normal(size=(UNITS_OF_HIDDEN_LAYER_1, UNITS_OF_HIDDEN_LAYER_2)).astype(np.float32)*math.sqrt(2/UNITS_OF_HIDDEN_LAYER_1)
            w3_initial = np.random.normal(size=(UNITS_OF_HIDDEN_LAYER_2, UNITS_OF_HIDDEN_LAYER_3)).astype(np.float32)*math.sqrt(2/UNITS_OF_HIDDEN_LAYER_2)
            w4_initial = np.random.normal(size=(UNITS_OF_HIDDEN_LAYER_3, NUMBER_OF_CLASSES)).astype(np.float32)

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

            #Layer 3
            w3 = tf.Variable(w3_initial)
            z3 = tf.matmul(l2, w3)
            bn3 = batch_norm_wrapper(z3, True)
            l3 = tf.nn.relu(bn3)

            # Softmax
            w4 = tf.Variable(w4_initial)
            b4 = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))
            predicted = tf.nn.tanh(tf.matmul(l3, w4) + b4)

            rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(price, predicted)))
            train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(rmse)
            #--------------------------------------------------------------------------------------------------------------#
            for v in tf.trainable_variables():
                sess.run(tf.variables_initializer([v]))
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            for i in range(STEP_NUMBER):
                sample = np.random.randint(TRAIN_NUMBER, size=2)
                batch_xs = train_input[sample][:]
                batch_ys = train_target[sample][:]
                sess.run(train_step, feed_dict={input: batch_xs, price: batch_ys})
                if i == STEP_NUMBER-1:
                    predicted_price.append(sess.run([predicted], feed_dict={input: test_input, price: test_target})[0])

            if(LOG_RETURN_USED):
                predicted_ = predicted_price[0]
                price_ = test_target
            else:
                predicted_ = scaler.inverse_transform(predicted_price[0])
                price_ = scaler.inverse_transform(test_target)

            total_cost = total_cost + math.sqrt(mean_squared_error(price_, predicted_))
            print_results(predicted_, price_, test_years, test_days)
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
        train_year = np.asarray(x_train[:, -1]).reshape(-1, 1)
        test_year = np.asarray(x_test[:, -1]).reshape(-1, 1)

        train_days = np.asarray(x_train[:, -2]).reshape(-1, 1)
        test_days = np.asarray(x_test[:, -2]).reshape(-1, 1)

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

def train_test_split_(data, slide_length_for_train, slide_length_for_test):
    start_index = 0
    end_index = 0
    train_list = list()
    test_list = list()
    while ((end_index + slide_length_for_test) < data.shape[0]):
        end_index = start_index + slide_length_for_train
        train_list.append(data[start_index:end_index])
        test_list.append(data[end_index:end_index + slide_length_for_test])
        start_index = start_index + slide_length_for_test
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

def initialize_setting(window_size, prediction_horizon, is_price_of_previous_days_allowed, slide_length_for_train, slide_length_for_test):
    data = preprocess_data(window_size, prediction_horizon, is_price_of_previous_days_allowed, slide_length_for_train)
    train_list, test_list = train_test_split_(data, slide_length_for_train, slide_length_for_test)
    x_train_list, x_test_list, train_year_list, test_year_list, train_list_days, test_list_days = exclude_days(train_list, test_list)
    input_number, train_input_list, train_target_list, test_input_list, test_target_list = split_input_target(x_train_list, x_test_list)

    return input_number, train_input_list, train_target_list, test_input_list, test_target_list, train_year_list, test_year_list, train_list_days, test_list_days

is_price_of_previous_days_allowed = True
for slide_length_for_train in [15]:
    for slide_length_for_test in [1]:
        for horizon in range(1, 5):
            for window in [3]:
                print('window: ', window,
                      "horizon:", horizon,
                      "slide_length_for_train:", slide_length_for_train,
                      "slide_length_for_test:", slide_length_for_test,
                      "is_price_of_previous_days_allowed:", is_price_of_previous_days_allowed)
                input_number, train_input_list, train_target_list, test_input_list, test_target_list, train_year_list, test_year_list, train_list_days, test_list_days = initialize_setting(
                    window, horizon, is_price_of_previous_days_allowed, slide_length_for_train, slide_length_for_test)
                run_print_model(input_number, train_input_list, train_target_list, test_input_list, test_target_list,
                                train_year_list, test_year_list, train_list_days, test_list_days)
