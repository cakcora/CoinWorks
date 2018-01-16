







import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np, tensorflow as tf, tqdm
from sklearn.preprocessing import OneHotEncoder



DECREASING_PRICE_LABEL = 0
INCREASING_PRICE_LABEL = 1
NUMBER_OF_CLASSES = 2


ROW = -1
COLUMN = -1


PRICED_BITCOIN_FILE_PATH = "C:\\Users\\nca150130\\Desktop\\matrix\\pricedBitcoin.csv"
DAILY_OCCURRENCE_FILE_PATH = "C:\\Users\\nca150130\\Desktop\\matrix\\dailyOccmatrices\\"


def preprocessData():
    priced_bitcoin = pd.read_csv(PRICED_BITCOIN_FILE_PATH, sep=",")
    priced_bitcoin_label = priced_bitcoin[['price']]
    priced_bitcoin_input = priced_bitcoin[['year', 'day', 'totaltx']]

# update labels based on increasing or decreasing bitcoin value
    temp = priced_bitcoin_label.values[0]
    for i, row in priced_bitcoin_label.itertuples():
        if temp < row:
            priced_bitcoin_label.ix[i, 'price'] = INCREASING_PRICE_LABEL
        else:
            priced_bitcoin_label.ix[i, 'price'] = DECREASING_PRICE_LABEL
        temp = row

# get normalized occurrence matrix in a flat format and merge with totaltx
    daily_occcurence_input = np.array([], dtype=np.float32)
    for index, row in priced_bitcoin_input.iterrows():
        daily_occurrence_matrix_path_name = DAILY_OCCURRENCE_FILE_PATH + "occ" + str(row['year']) + 'day' + '{:03}'.format(row['day']) + ".csv"
        daily_occurrence_matrix = pd.read_csv(daily_occurrence_matrix_path_name, sep=",", header=None).values
        daily_occurrence_normalized_matrix = np.asarray(daily_occurrence_matrix).reshape(1, daily_occurrence_matrix.size)/row['totaltx']
        if daily_occcurence_input.size==0:
            daily_occcurence_input = daily_occurrence_normalized_matrix
        else:
            daily_occcurence_input = np.concatenate((daily_occcurence_input, daily_occurrence_normalized_matrix), axis=0)

    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    totaltx_scaled = min_max_scaler.fit_transform(np.asarray(priced_bitcoin_input['totaltx']).reshape(-1,1))
    input = np.concatenate((daily_occcurence_input, totaltx_scaled), axis=1)

    return input, priced_bitcoin_label.values

input, label = preprocessData()
np.savetxt("C:\\Users\\nca150130\\PycharmProjects\\CoinWorks\\src\\main\\python\\input.txt", input)
np.savetxt("C:\\Users\\nca150130\\PycharmProjects\\CoinWorks\\src\\main\\python\\label.txt", label)
