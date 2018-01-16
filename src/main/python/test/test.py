







import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np, tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time


start_time = time.time()
#my_code
elapsed_time=time.time()-start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


DECREASING_PRICE_LABEL = 0
INCREASING_PRICE_LABEL = 1
NUMBER_OF_CLASSES = 1


ROW = -1
COLUMN = -1
TEST_SPLIT = 0.1


PRICED_BITCOIN_FILE_PATH = "C:\\Users\\nca150130\\Desktop\\matrix\\pricedBitcoin.csv"
DAILY_OCCURRENCE_FILE_PATH = "C:\\Users\\nca150130\\Desktop\\matrix\\dailyOccmatrices\\"


#DEEP_LEARNING_PARAMETERS
LEARNING_RATE = 0.01
BATCH_SIZE = 20
STEP_NUMBER = 50000
UNITS_OF_HIDDEN_LAYER_1 = 128
UNITS_OF_HIDDEN_LAYER_2 = 64
EPSILON = 1e-3
INPUT_NUMBER = -1

#####BITCOIN_PREDICTION_PARAMETERS#####
#ALL YEAR OR YEAR BASED PREDICTION
ALL_YEAR_INPUT_ALLOWED = False
YEAR = 2012
#AGGREGATION AND WINDOW SIZE FOR PREVIOUS DAYS
AGGREGATION_OF_PREVIOUS_DAYS_ALLOWED = False
WINDOW_SIZE = -1
IS_PRICE_OF_PREVIOUS_DAYS_ALLOWED = False
#HORIZON AND WINDOW SIZE
HORIZON_ALLOWED = False
PREDICTION_HORIZON = 1

#todo 1: print false positive, false negative
#todo 2: change decay parameter
#todo 3: output function prediction price, real price, difference between them with normalized version

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

def build_graph(is_training):
    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NUMBER])
    y_ = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])

    # Layer 1
    w1 = tf.Variable(w1_initial)
    z1 = tf.matmul(x,w1)
    bn1 = batch_norm_wrapper(z1, is_training)
    l1 = tf.nn.relu(bn1)

    #Layer 2
    w2 = tf.Variable(w2_initial)
    z2 = tf.matmul(l1,w2)
    bn2 = batch_norm_wrapper(z2, is_training)
    l2 = tf.nn.relu(bn2)

    # Softmax
    w3 = tf.Variable(w3_initial)
    b3 = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))
    y  = tf.matmul(l2, w3) + b3

    # Loss, Optimizer and Predictions
    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y, y_))))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(rmse)
    return (x, y_), train_step, rmse, y, tf.train.Saver()

def merge_data(occurrence_data, daily_occurrence_normalized_matrix):
    if(AGGREGATION_OF_PREVIOUS_DAYS_ALLOWED):
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

def get_daily_occurrence_matrices(priced_bitcoin, current_row):
    previous_price_data = np.array([], dtype=np.float32)
    occurrence_data = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin.iterrows():
        if not ((row.values == current_row.values).all()):
            previous_price_data = np.append(previous_price_data, row['price'])
            daily_occurrence_normalized_matrix = get_normalized_matrix_from_file(row['day'], row['year'], row['totaltx'])
            occurrence_data = merge_data(occurrence_data, daily_occurrence_normalized_matrix)
    if(IS_PRICE_OF_PREVIOUS_DAYS_ALLOWED):
        occurrence_data = np.concatenate((occurrence_data, np.asarray(previous_price_data).reshape(1,-1)), axis=1)
    occurrence_input = np.concatenate((occurrence_data, np.asarray(current_row['price']).reshape(1,1)), axis=1)
    return occurrence_input

def get_normalized_matrix_from_file(day, year, totaltx):
    daily_occurrence_matrix_path_name = DAILY_OCCURRENCE_FILE_PATH + "occ" + str(year) + 'day' + '{:03}'.format(day) + ".csv"
    daily_occurrence_matrix = pd.read_csv(daily_occurrence_matrix_path_name, sep=",", header=None).values
    return np.asarray(daily_occurrence_matrix).reshape(1, daily_occurrence_matrix.size)/totaltx

def preprocess_data():
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_PATH, sep=",")
    if (ALL_YEAR_INPUT_ALLOWED):
        pass
    else:
        priced_bitcoin = priced_bitcoin[priced_bitcoin['year']==YEAR].reset_index(drop=True)

# get normalized occurrence matrix in a flat format and merge with totaltx
    daily_occurrence_input = np.array([], dtype=np.float32)
    temp = np.array([], dtype=np.float32)
    for current_index, current_row in priced_bitcoin.iterrows():
        if(current_index<(WINDOW_SIZE+PREDICTION_HORIZON-1)):
            pass
        else:
            start_index = current_index-(WINDOW_SIZE + PREDICTION_HORIZON) + 1
            end_index = current_index-PREDICTION_HORIZON
            temp = get_daily_occurrence_matrices(priced_bitcoin[start_index:end_index+1], current_row)
        if daily_occurrence_input.size == 0:
            daily_occurrence_input = temp
        else:
            daily_occurrence_input = np.concatenate((daily_occurrence_input, temp), axis=0)

    return daily_occurrence_input

def run_model(train_input, train_target, test_input, test_target):
    TRAIN_NUMBER = x_train.shape[0]
    # Placeholders
    x = tf.placeholder(tf.float32, shape=[None, INPUT_NUMBER])
    y_ = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    (x, y_), train_step, rsme, y, saver = build_graph(is_training=True)

    cost_list = []
    predicted_price = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(STEP_NUMBER):
            sample = np.random.randint(TRAIN_NUMBER, size=BATCH_SIZE)
            batch_xs = train_input[sample][:]
            batch_ys = train_target[sample][:]
            train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
            if i % 50 is 0:
                res = sess.run([rsme],feed_dict={x: test_input, y_: test_target})
                cost_list.append(res[0])
            if i == STEP_NUMBER-1:
                predicted_price.append(sess.run([y], feed_dict={x: test_input, y_: test_target})[0])
    return cost_list, predicted_price, test_target
#----------------------------------------------------------------------------------------------------------------------#
WINDOW_SIZE = 1
PREDICTION_HORIZON = 1
data = preprocess_data()
x_train, x_test = train_test_split(data, test_size=TEST_SPLIT)

row, column = x_train.shape
INPUT_NUMBER = column-1
train_target = np.asarray(x_train[:,-1]).reshape(-1,1)
train_input = x_train[:,0:column-1]

test_target = np.asarray(x_test[:,-1]).reshape(-1,1)
test_input = x_test[:,0:column-1]

scaler = preprocessing.StandardScaler()
scaled_train_totalx = scaler.fit_transform(np.asarray(train_input[:,-1]).reshape(-1,1))
scaled_test_totalx = scaler.fit_transform(np.asarray(test_input[:,-1]).reshape(-1,1))

train_input[:,-1] = np.asarray(scaled_train_totalx).reshape(-1,)
test_input[:,-1] = np.asarray(scaled_test_totalx).reshape(-1,)

w1_initial = np.random.normal(size=(INPUT_NUMBER, UNITS_OF_HIDDEN_LAYER_1)).astype(np.float32)
w2_initial = np.random.normal(size=(UNITS_OF_HIDDEN_LAYER_1, UNITS_OF_HIDDEN_LAYER_2)).astype(np.float32)
w3_initial = np.random.normal(size=(UNITS_OF_HIDDEN_LAYER_2, NUMBER_OF_CLASSES)).astype(np.float32)

cost, predicted, price = run_model(train_input, train_target, test_input, test_target)




