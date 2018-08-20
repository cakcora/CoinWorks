import numpy as np
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np, tensorflow as tf
import math
from sklearn.model_selection import train_test_split
import sys
import os


LEARNING_RATE = 0.0010
STEP_NUMBER = 10000
UNITS_OF_HIDDEN_LAYER_1 = 16
UNITS_OF_HIDDEN_LAYER_2 = 8
UNITS_OF_HIDDEN_LAYER_3 = 4
DISPLAY_STEP = int(STEP_NUMBER / 10)
NUMBER_OF_CLASSES = 1


LOG_RETURN = True
WINDOW = -1
HORIZON = -1
TEST_SPLIT = 0.2


def build_graph(input_number):

    # Placeholders
    input = tf.placeholder(tf.float32, shape=[None, input_number])
    price = tf.placeholder(tf.float32, shape=[None, NUMBER_OF_CLASSES])

    # Layer 1
    w1 = tf.Variable(tf.random_uniform([input_number, UNITS_OF_HIDDEN_LAYER_1], minval=-math.sqrt(2/(input_number+UNITS_OF_HIDDEN_LAYER_1)), maxval=math.sqrt(2/(input_number+UNITS_OF_HIDDEN_LAYER_1))))
    z1 = tf.matmul(input, w1)
    l1 = tf.nn.relu(z1)
    l1_dropout = tf.nn.dropout(l1, 0.8)

    # Layer 2
    w2 = tf.Variable(tf.random_uniform([UNITS_OF_HIDDEN_LAYER_1, UNITS_OF_HIDDEN_LAYER_2], minval=-math.sqrt(2/(UNITS_OF_HIDDEN_LAYER_1+UNITS_OF_HIDDEN_LAYER_2)), maxval=math.sqrt(2/(UNITS_OF_HIDDEN_LAYER_1+UNITS_OF_HIDDEN_LAYER_2))))
    z2 = tf.matmul(l1_dropout, w2)
    l2 = tf.nn.relu(z2)
    l2_dropout = tf.nn.dropout(l2, 0.8)

    # Layer 3
    w3 = tf.Variable(tf.random_uniform([UNITS_OF_HIDDEN_LAYER_2, UNITS_OF_HIDDEN_LAYER_3], minval=-math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_2+UNITS_OF_HIDDEN_LAYER_3)), maxval=math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_2+UNITS_OF_HIDDEN_LAYER_3))))
    z3 = tf.matmul(l2_dropout, w3)
    l3 = tf.nn.tanh(z3)
    l3_dropout = tf.nn.dropout(l3, 0.8)

    # Linear
    w4 = tf.Variable(tf.random_uniform([UNITS_OF_HIDDEN_LAYER_3, NUMBER_OF_CLASSES], minval=-math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_3+NUMBER_OF_CLASSES)), maxval=math.sqrt(6/(UNITS_OF_HIDDEN_LAYER_3+NUMBER_OF_CLASSES))))
    b4 = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))
    predicted = tf.matmul(l3_dropout, w4) + b4

    rmse = tf.losses.mean_squared_error(price, predicted)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(rmse)

    return (input, price), train_step, predicted, rmse, tf.train.Saver()


def print_results(test_target, predicted_list, predicted_day_list, predicted_year_list, result_file):
    myFile = open(result_file, 'a')

    if(LOG_RETURN):
        pass
    else:
        test_target = target_scaler.inverse_transform(test_target)
        predicted_list = target_scaler.inverse_transform(predicted_list)
    prefix = str(True) + "\t" + str(True) + '\t' + str(WINDOW) + '\t' + str(HORIZON)

    for pred, real, day, year in zip(test_target, predicted_list, predicted_day_list, predicted_year_list):
        myFile.write(prefix + "\t" +
                     str(300) + "\t" +
                     str(1) + "\t" +
                     str(pred[0]) + "\t" +
                     str(real[0]) + "\t" +
                     str(int(day[0])) + "\t" +
                     str(int(year[0])) + '\n')
    cost = math.sqrt(mean_squared_error(test_target, predicted_list))
    print("COST: ", cost)
    myFile.close()


def get_file_parameters():

    filtrated_log_return_file = sys.argv[1]
    result_file = sys.argv[2]

    #remove existing file
    if os.path.isfile(result_file):
        os.remove(result_file)

    return filtrated_log_return_file, result_file


filtrated_log_return_file, result_file = get_file_parameters()


filtrated_log_returns = pd.read_csv(filtrated_log_return_file, sep="\t", header=None)
filtrated_log_returns.columns = ['priced', 'aggregated', "window", "horizon", "threshold", "training_length", "test_length", "predicted", "real", "year", "day"]


for WINDOW in [3,5]:
    for HORIZON in [1,2,5,10]:

        input = filtrated_log_returns[(filtrated_log_returns['window'] == WINDOW) & (filtrated_log_returns['horizon'] == HORIZON)]

        threshold_0 = np.asarray(input[input["threshold"] == 0]["predicted"].values).reshape(-1,1)
        threshold_10 = np.asarray(input[input["threshold"] == 10]["predicted"].values).reshape(-1,1)
        threshold_20 = np.asarray(input[input["threshold"] == 20]["predicted"].values).reshape(-1,1)
        threshold_30 = np.asarray(input[input["threshold"] == 30]["predicted"].values).reshape(-1,1)
        threshold_40 = np.asarray(input[input["threshold"] == 40]["predicted"].values).reshape(-1,1)
        threshold_50 = np.asarray(input[input["threshold"] == 50]["predicted"].values).reshape(-1,1)
        threshold_60 = np.asarray(input[input["threshold"] == 60]["predicted"].values).reshape(-1,1)
        threshold_70 = np.asarray(input[input["threshold"] == 70]["predicted"].values).reshape(-1,1)
        threshold_80 = np.asarray(input[input["threshold"] == 80]["predicted"].values).reshape(-1,1)
        threshold_90 = np.asarray(pd.DataFrame(input[input["threshold"] == 90], columns=["predicted", "real", "year", "day"]).values).reshape(-1,4)


        concat_10 = np.concatenate((threshold_0, threshold_10), axis=1)
        concat_30 = np.concatenate((threshold_20, threshold_30), axis=1)
        concat_50 = np.concatenate((threshold_40, threshold_50), axis=1)
        concat_70 = np.concatenate((threshold_60, threshold_70), axis=1)
        concat_90 = np.concatenate((threshold_80, threshold_90), axis=1)

        concat_30 = np.concatenate((concat_10, concat_30), axis=1)
        concat_50 = np.concatenate((concat_30, concat_50), axis=1)
        concat_70 = np.concatenate((concat_50, concat_70), axis=1)
        data = np.concatenate((concat_70, concat_90), axis=1)
        data = pd.DataFrame(data, columns=["t0", "t10", "t20", "t30", "t40", "t50", "t60", "t70", "t80", "t90", "real", "year", "day"])

        if(LOG_RETURN):
            pass
        else:
            input_scaler = preprocessing.MinMaxScaler()
            data[["t0", "t10", "t20", "t30", "t40", "t50", "t60", "t70", "t80", "t90"]] = input_scaler.fit_transform(data[["t0", "t10", "t20", "t30", "t40", "t50", "t60", "t70", "t80", "t90"]])

            target_scaler = preprocessing.MinMaxScaler()
            data[["real"]] = target_scaler.fit_transform(data[["real"]])


        train, test = train_test_split(data, test_size=TEST_SPLIT, random_state=42)


        train_input = pd.DataFrame(train, columns=["t0", "t10", "t20", "t30", "t40", "t50", "t60", "t70", "t80", "t90"]).values
        train_target = pd.DataFrame(train, columns=["real"]).values

        test_input = pd.DataFrame(test, columns=["t0", "t10", "t20", "t30", "t40", "t50", "t60", "t70", "t80", "t90"]).values
        test_target = pd.DataFrame(test, columns=["real", "year", "day"]).values


        input_number = train_input.shape[1]

        tf.reset_default_graph()
        (input, price), train_step, predicted, loss, saver = build_graph(input_number)

        train_loss_list = []
        test_loss_list = []
        predicted_list = []
        predicted_day_list = []
        predicted_year_list = []
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
                        test_loss_list.append(sess.run([loss], feed_dict={input: test_input, price: np.asarray(test_target[:,0]).reshape(-1,1)})[0])
                        predicted_list.append(sess.run([predicted], feed_dict={input: test_input, price: np.asarray(test_target[:,0]).reshape(-1,1)})[0])
                        predicted_day_list.append(test_target[:,-1])
                        predicted_year_list.append(test_target[:,-2])

            print_results(np.asarray(test_target[:,0]).reshape(-1,1), np.asarray(predicted_list[0]).reshape(-1,1), np.asarray(predicted_day_list[0]).reshape(-1,1), np.asarray(predicted_year_list[0]).reshape(-1,1), result_file)
