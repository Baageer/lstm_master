import tushare as ts
import numpy as np
import pickle
from sklearn import preprocessing

# pro = ts.pro_api('ed13f008c584df320f79a1b4295ba81f132df54a2b72cad256fe87f4')

# df = ts.pro_bar(ts_code='600219.SH',  start_date='20180101', end_date='20180511', 
#                  ma=[5, 20, 50] )
# print(df)

# t = input()


savafilename = "srcdata5_13"

s_list = ['600297', '600170', '600219', '600369', '600372',
          ]#'600486', '600511', '600583', '600612', '600132']

train_length_list = []
test_length_list = []
#print(np.array(data.values.tolist()))
#print(np.array(data['close'].values.tolist()))

trainx = []
trainy = []
testx = []
testy = []
data_dict = {}
for item in s_list:
    data = ts.get_hist_data(item, start="2018-01-01", end="2019-12-31")
    
    data = data.reindex(index=data.index[::-1])

    data1 = ts.get_hist_data(item, start="2020-01-02", end="2020-04-15")
    data1 = data1.reindex(index=data1.index[::-1])

    trainx += data.values.tolist()
    trainy += data['close'].values.tolist()
    train_length_list.append(len(data.values.tolist()))

    testx += data1.values.tolist()
    testy += data1['close'].values.tolist()
    test_length_list.append(len(data1.values.tolist()))

print(train_length_list)
print(test_length_list)

TIME_STEPS = 13     # backpropagation through time 的 time_steps
BATCH_SIZE = 256

ss_x = preprocessing.StandardScaler()
train_x = ss_x.fit_transform(np.array(trainx))
test_x = ss_x.fit_transform(np.array(testx))
ss_y = preprocessing.StandardScaler()
train_y = ss_y.fit_transform(np.array(trainy).reshape(-1, 1))
test_y = ss_y.fit_transform(np.array(testy).reshape(-1, 1))

train_start = 0
test_start = 0
save_train = []
save_test = []
for idx,item in enumerate(s_list):
    train_length = train_length_list[idx]
    test_length = test_length_list[idx]

    train_dataset_x = train_x[train_start:train_start+train_length]
    train_dataset_y = train_y[train_start:train_start+train_length]

    test_dataset_x = test_x[test_start:test_start+test_length]
    test_dataset_y = test_y[test_start:test_start+test_length]

    idx_start = 0
    while(True):
        temp_x = train_dataset_x[idx_start:idx_start+TIME_STEPS]
        temp_y = train_dataset_y[idx_start+TIME_STEPS]

        save_train.append({"x":temp_x, "y":temp_y})

        idx_start += 1
        if idx_start+TIME_STEPS == train_length:
            break
    idx_start = 0
    while(True):
        temp_x = test_dataset_x[idx_start:idx_start+TIME_STEPS]
        temp_y = test_dataset_y[idx_start+TIME_STEPS]

        save_test.append({"x":temp_x, "y":temp_y})

        idx_start += 1
        if idx_start+TIME_STEPS == test_length:
            break

    train_start += train_length
    test_start += test_length

print(len(save_train))

# data_dict['train_x'] = np.array(trainx)
# data_dict['train_y'] = np.array(trainy)


# data_dict['test_x'] = np.array(testx)
# data_dict['test_y'] = np.array(testy)

data_dict['train'] = save_train
data_dict['test'] = save_test

f = open(savafilename, 'wb')
pickle.dump(data_dict, f)
f.close()
