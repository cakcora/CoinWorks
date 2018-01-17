
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np, tensorflow as tf
import os
import gc
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

METHOD = "amo"
PRICED_BITCOIN_FILE_PATH = "D:\\Bitcoin\\pricedBitcoin2009-2018.csv"
DAILY_OCCURRENCE_FILE_PATH = "D:\\Bitcoin\\createddata\\daily" + METHOD + "matrices\\"
RESULT_FOLDER = "D:\\Bitcoin\\createddata\\results\\"
RESULT_FILE = METHOD + 'bitcoin_prices_prediction.csv'

# clean the result file.

if os.path.isfile(RESULT_FOLDER + RESULT_FILE):
    os.remove(RESULT_FOLDER + RESULT_FILE)

NUMBER_OF_CLASSES = 1

ROW = -1
COLUMN = -1
TEST_SPLIT = 0.2

#DEEP_LEARNING_PARAMETERS
REGULARIZATION_FOR_LOG = 0.0000001
LEARNING_RATE = 0.01
BATCH_SIZE = 20
STEP_NUMBER = 100000
UNITS_OF_HIDDEN_LAYER_1 = 128
UNITS_OF_HIDDEN_LAYER_2 = 64
EPSILON = 1e-3
DISPLAY_STEP = int(STEP_NUMBER / 10)
ALL_YEAR_INPUT_ALLOWED = False
YEAR = 2017

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
        occurrence_data = np.concatenate((occurrence_data, np.asarray(previous_price_data).reshape(1,-1)), axis=1)
    occurrence_input = np.concatenate((occurrence_data, np.asarray(current_row['price']).reshape(1,1)), axis=1)
    occurrence_input = np.concatenate((occurrence_input, np.asarray(current_row['day']).reshape(1,1)), axis=1)
    return occurrence_input

def get_normalized_matrix_from_file(day, year, totaltx):
    daily_occurrence_matrix_path_name = DAILY_OCCURRENCE_FILE_PATH + METHOD + str(year) + '{:03}'.format(day) + ".csv"
    daily_occurrence_matrix = pd.read_csv(daily_occurrence_matrix_path_name, sep=",", header=None).values
    return np.asarray(daily_occurrence_matrix).reshape(1, daily_occurrence_matrix.size)/totaltx

def preprocess_data(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_PATH, sep=",")
    if (ALL_YEAR_INPUT_ALLOWED):
        pass
    else:
        priced_bitcoin = priced_bitcoin[priced_bitcoin['year']==YEAR].reset_index(drop=True)

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


def print_model(test_input, predicted, test_target, test_days):
    myFile = open(RESULT_FOLDER + RESULT_FILE, 'a')

    original_log_return = np.log(np.asarray(test_target).reshape(-1,)/test_input[:,-1])
    predicted_log_return = np.log(np.asarray(predicted).reshape(-1,)/test_input[:,-1])
    predicted = np.asarray(predicted).flatten()

    for p, t, o_l, p_l, t_d in zip(predicted, test_target, original_log_return, predicted_log_return, test_days):
        myFile.write("test\t" +
                     str(YEAR) + "\t" + str(int(t_d[0])) + "\t" +
                     str(is_price_of_previous_days_allowed) + "\t" +
                     str(aggregation_of_previous_days_allowed) + "\t" +
                     str(window_size) + "\t" +
                     str(prediction_horizon) + "\t" +
                     str(p) + "\t" + str(t[0]) + "\t" + str(o_l) + "\t" + str(p_l) + '\n')

    myFile.close()

def build_graph(sess, is_training):
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
    bn1 = batch_norm_wrapper(z1, is_training)
    l1 = tf.nn.relu(bn1)

    #Layer 2
    w2 = tf.Variable(w2_initial)
    z2 = tf.matmul(l1, w2)
    bn2 = batch_norm_wrapper(z2, is_training)
    l2 = tf.nn.relu(bn2)

    # Softmax
    w3 = tf.Variable(w3_initial)
    b3 = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))
    predicted = tf.nn.relu(tf.matmul(l2, w3) + b3)

    original_log = tf.log(price/input[:, input_number-1]+REGULARIZATION_FOR_LOG)
    predicted_log = tf.log(predicted/input[:, input_number-1]+REGULARIZATION_FOR_LOG)

    # Loss, Optimizer and Predictions
    rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(original_log, predicted_log)))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(rmse)
    return (input, price), train_step, rmse, predicted, original_log, predicted_log, tf.train.Saver()


def run_print_model(input_number, train_input, train_target, test_input, test_target, test_days):
    #(input, price), train_step, rmse, predicted, original_log, predicted_log, saver = build_graph(sess, is_training=True)
    train_cost = []
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

        #original_log = tf.log(price/tf.reshape(input[:, input_number-1], [-1,1]) + REGULARIZATION_FOR_LOG)
        #predicted_log = tf.log(predicted/tf.reshape(input[:, input_number-1], [-1,1]) + REGULARIZATION_FOR_LOG)

        # Loss, Optimizer and Predictions
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

            if i % DISPLAY_STEP is 0:
                print("i: ", i, " cost: ", sess.run([rmse], feed_dict={input: batch_xs, price: batch_ys}))
                #print("original_log: ", sess.run([original_log], feed_dict={input: batch_xs, price: batch_ys})[0])
                #print("predicted_log: ", sess.run([predicted_log], feed_dict={input: batch_xs, price: batch_ys})[0])
            if i == STEP_NUMBER-1:
                predicted_price.append(sess.run([predicted], feed_dict={input: test_input, price: test_target})[0])

    print_model(test_input, predicted_price, test_target, test_days)
#----------------------------------------------------------------------------------------------------------------------#
parameter_dict = {#0: dict({'is_price_of_previous_days_allowed':True, 'aggregation_of_previous_days_allowed':True})}
    1: dict({'is_price_of_previous_days_allowed': True, 'aggregation_of_previous_days_allowed': False})}

def exclude_days(train, test):
    row, column = train.shape
    train_days = np.asarray(train[:, -1]).reshape(-1, 1)
    x_train = train[:, 0:column - 1]
    test_days = np.asarray(test[:, -1]).reshape(-1, 1)
    x_test = test[:, 0:column - 1]

    return x_train, x_test, train_days, test_days

def initialize_setting(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed):
    data = preprocess_data(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
    train, test = train_test_split(data, test_size=TEST_SPLIT)
    x_train, x_test, train_days, test_days = exclude_days(train, test)
    row, column = x_train.shape
    input_number = column - 1
    train_target = np.asarray(x_train[:, -1]).reshape(-1, 1)
    train_input = x_train[:, 0:column - 1]
    test_target = np.asarray(x_test[:, -1]).reshape(-1, 1)
    test_input = x_test[:, 0:column - 1]
    return input_number, train_input, train_target, test_input, test_target, train_days, test_days

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
            input_number, train_input, train_target, test_input, test_target, train_days, test_days = initialize_setting(window_size, prediction_horizon, is_price_of_previous_days_allowed, aggregation_of_previous_days_allowed)
            run_print_model(input_number, train_input, train_target, test_input, test_target, test_days)





