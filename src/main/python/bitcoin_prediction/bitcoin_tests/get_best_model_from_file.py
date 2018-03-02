






import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np, tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import math

FILE_NAME = ",,\\PycharmProjects\\CoinWorks\\src\\main\\python\\results\\rnn_order_2017.csv"
MIN_COST = 10000

with open(FILE_NAME, "r") as ins:

    for line in ins:
        cost = 0
        counter = 0
        while ":" in line:
            print(line, end='', flush=True)
            line = ins.readline()
        cost = 0
        while ":" not in line:
            counter = counter + 1.0
            splitted = line.split(",")
            cost = cost + math.fabs(float(splitted[4])-float(splitted[3]))
            line = ins.readline()
        cost = cost/counter
        if cost < MIN_COST:
            MIN_COST = cost
            print("COST: ", cost)
        if ":" in line:
            print(line, end='', flush=True)
print("bitti")
