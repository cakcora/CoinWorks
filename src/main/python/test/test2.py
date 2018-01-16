


import numpy as np





data = np.random.sample(192)
train_split = 0.8
batch_size_div = 10


data_size = int(len(data))
train_size_initial = int(data_size * train_split)
x_samples = data[-data_size:, :]


if train_size_initial < batch_size_div:
    batch_size = 1
else:
    batch_size = int(train_size_initial / batch_size_div)

train_size = int(int(train_size_initial / batch_size) * batch_size)  # provide even division of training / batches
val_size = int(int((data_size - train_size) / batch_size) * batch_size)  # provide even division of val / batches
print('Data Size: {}  Train Size: {}   Batch Size: {}'.format(data_size, train_size, batch_size))


train, val = x_samples[0:train_size, 0:-1], x_samples[train_size:train_size + val_size, 0:-1]
