



import numpy as np





data = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40])
SLIDING_BATCH_SIZE = 10


start_index = 0
end_index = 0
train_list = list()
test_list = list()
while((end_index+SLIDING_BATCH_SIZE) < data.shape[0]):
    end_index = end_index + SLIDING_BATCH_SIZE
    train_list.append(data[start_index:end_index])
    test_list.append(data[end_index:end_index+SLIDING_BATCH_SIZE])
    start_index = start_index + SLIDING_BATCH_SIZE



for i in range(0, len(train_list)):
    print(train_list[i])


print("---------------------------------------------------------")


for i in range(0, len(test_list)):
    print(test_list[i])
