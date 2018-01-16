







import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np, tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


DECREASING_PRICE_LABEL = 0
INCREASING_PRICE_LABEL = 1
NUMBER_OF_CLASSES = 2


ROW = -1
COLUMN = -1
TEST_SPLIT = 0.1


PRICED_BITCOIN_FILE_PATH = "C:\\Users\\nca150130\\Desktop\\matrix\\pricedBitcoin.csv"
DAILY_OCCURRENCE_FILE_PATH = "C:\\Users\\nca150130\\Desktop\\matrix\\dailyOccmatrices\\"


LEARNING_RATE = 0.01
BATCH_SIZE = 20
STEP_NUMBER = 50000
UNITS_OF_HIDDEN_LAYER_1 = 128
UNITS_OF_HIDDEN_LAYER_2 = 64
EPSILON = 1e-3
INPUT_NUMBER = 401


AGGREGATION_OF_PREVIOUS_DAYS_ALLOWED = True
WINDOW_SIZE = 3
ALL_YEAR_INPUT_ALLOWED = False
YEAR = 2016


#todo 1: k-window based on day aggregation
#todo 2: k-window based on day momentum
#todo 3: horizon (based on previous days predict kth day price)
#todo 4: print false positive, false negative
#todo 5: change decay parameter

w1_initial = np.random.normal(size=(INPUT_NUMBER, UNITS_OF_HIDDEN_LAYER_1)).astype(np.float32)
w2_initial = np.random.normal(size=(UNITS_OF_HIDDEN_LAYER_1, UNITS_OF_HIDDEN_LAYER_2)).astype(np.float32)
w3_initial = np.random.normal(size=(UNITS_OF_HIDDEN_LAYER_2, NUMBER_OF_CLASSES)).astype(np.float32)

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
    l1 = tf.nn.sigmoid(bn1)

    #Layer 2
    w2 = tf.Variable(w2_initial)
    z2 = tf.matmul(l1,w2)
    bn2 = batch_norm_wrapper(z2, is_training)
    l2 = tf.nn.sigmoid(bn2)

    # Softmax
    w3 = tf.Variable(w3_initial)
    b3 = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))
    y  = tf.matmul(l2, w3) + b3

    # Loss, Optimizer and Predictions
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.arg_max(y,1),tf.arg_max(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    return (x, y_), train_step, accuracy, y, tf.train.Saver()

def get_daily_occurrence_matrices(row):
    window_size_index = WINDOW_SIZE
    previous_day_data = np.array([], dtype=np.float32)
    while(window_size_index!=0):
        window_size_index = window_size_index-1
        previous_day = row['day'] - window_size_index
        daily_occurrence_matrix_path_name = DAILY_OCCURRENCE_FILE_PATH + "occ" + str(row['year']) + 'day' + '{:03}'.format(previous_day) + ".csv"
        daily_occurrence_matrix = pd.read_csv(daily_occurrence_matrix_path_name, sep=",", header=None).values
        daily_occurrence_normalized_matrix = np.asarray(daily_occurrence_matrix).reshape(1, daily_occurrence_matrix.size)/row['totaltx']
        if(AGGREGATION_OF_PREVIOUS_DAYS_ALLOWED):
            if previous_day_data.size==0:
                previous_day_data = daily_occurrence_normalized_matrix
            else:
                previous_day_data = np.add(previous_day_data, daily_occurrence_normalized_matrix)
        else:
            if previous_day_data.size==0:
                previous_day_data = daily_occurrence_normalized_matrix
            else:
                previous_day_data = np.concatenate((previous_day_data, daily_occurrence_normalized_matrix), axis=0)
    return previous_day_data


def preprocess_data():
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_PATH, sep=",")
    (ROW_NUMBER, COLUMN_NUMBER) = priced_bitcoin.shape
    if (ALL_YEAR_INPUT_ALLOWED):
        pass
    else:
        priced_bitcoin = priced_bitcoin[priced_bitcoin['year']==YEAR].reset_index(drop=True)
    priced_bitcoin_label = priced_bitcoin[['price']].reset_index(drop=True)
    priced_bitcoin_input = priced_bitcoin[['year', 'day', 'totaltx']].reset_index(drop=True)

# update labels based on increasing or decreasing bitcoin value
    temp = priced_bitcoin_label.values[0]
    for i, row in priced_bitcoin_label.itertuples():
        if temp < row:
            priced_bitcoin_label.ix[i, 'price'] = INCREASING_PRICE_LABEL
        else:
            priced_bitcoin_label.ix[i, 'price'] = DECREASING_PRICE_LABEL
        temp = row

# get normalized occurrence matrix in a flat format and merge with totaltx
    daily_occurence_input = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin_input.iterrows():
        if(index<(WINDOW_SIZE-1)):
            pass
        else:
            daily_occurence_matrices = get_daily_occurrence_matrices(row)
            if daily_occurence_input.size == 0:
                daily_occurence_input = daily_occurence_matrices
            else:
                daily_occurence_input = np.concatenate((daily_occurence_input, daily_occurence_matrices), axis=0)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    totaltx_scaled = min_max_scaler.fit_transform(np.asarray(priced_bitcoin_input['totaltx']).reshape(-1,1)[(WINDOW_SIZE-1):ROW_NUMBER,:])
    input = np.concatenate((daily_occurence_input, totaltx_scaled), axis=1)

    return input, priced_bitcoin_label.values[(WINDOW_SIZE-1):ROW_NUMBER,:]

input, label = preprocess_data()
x_train, x_test, y_train, y_test = train_test_split(input, label, test_size=TEST_SPLIT)

onehot_encoder = OneHotEncoder(sparse=False)
onehot_encoder.fit(label)
y_train = onehot_encoder.transform(y_train)
y_test = onehot_encoder.transform(y_test)

TRAIN_NUMBER = x_train.shape[0]
# Placeholders
x = tf.placeholder(tf.float32, shape=[None, INPUT_NUMBER])
y_ = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
(x, y_), train_step, accuracy, y, saver = build_graph(is_training=True)

acc = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(STEP_NUMBER):
        sample = np.random.randint(TRAIN_NUMBER, size=BATCH_SIZE)
        batch_xs = x_train[sample][:]
        batch_ys = y_train[sample][:]
        train_step.run(feed_dict={x: batch_xs, y_: batch_ys})
        if i % 50 is 0:
            res = sess.run([accuracy],feed_dict={x: x_test, y_: y_test})
            acc.append(res[0])

print("Final accuracy:", max(acc))

