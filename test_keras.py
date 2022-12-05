import data_processing as dp
import numpy as np
from keras.models import load_model

dp_test = dp.data_processing(['600219.SH','600170.SH','603799.SH', '600369.SH', '600372.SH',
                              '600893.SH','603288.SH','600570.SH', '601899.SH', '600837.SH',],
                         "2022-01-01", 
                         "2022-05-31",codefile="SZ50.txt")
batch_size = 128
n_step = 30
data_col = ['open','close','high','low','pct_chg','ma5','ma20', 'ma50', 'vol', 'amount']
input_size = len(data_col)

test_data = dp_test.data_prepare(n_step, 1, data_col=data_col, clean_tmp=False)

# x_test, y_test = list(zip(*test_data))
# x_test, y_test = np.array(x_test), np.array(y_test)

test_data0, test_data1, test_data2, test_data3 = [],[],[],[]
for data in test_data:
    if data[1][0] == 1:
        test_data0.append(data)
    elif data[1][1] == 1:
        test_data1.append(data)
    elif data[1][2] == 1:
        test_data2.append(data)
    elif data[1][3] == 1:
        test_data3.append(data)

x_test0, y_test0 = list(zip(*test_data0))
x_test0, y_test0 = np.array(x_test0), np.array(y_test0)
x_test1, y_test1 = list(zip(*test_data1))
x_test1, y_test1 = np.array(x_test1), np.array(y_test1)
x_test2, y_test2 = list(zip(*test_data2))
x_test2, y_test2 = np.array(x_test2), np.array(y_test2)
x_test3, y_test3 = list(zip(*test_data3))
x_test3, y_test3 = np.array(x_test3), np.array(y_test3)


model = load_model('model-2000.h5')

#评估
score0 = model.evaluate(x_test0, y_test0, 
                       batch_size = batch_size, 
                       verbose = 1)
print('loss0:',score0[0])
print('acc0:',score0[1])

score1 = model.evaluate(x_test1, y_test1, 
                       batch_size = batch_size, 
                       verbose = 1)
print('loss1:',score1[0])
print('acc1:',score1[1])

score2 = model.evaluate(x_test2, y_test2, 
                       batch_size = batch_size, 
                       verbose = 1)
print('loss2:',score2[0])
print('acc2:',score2[1])

score3 = model.evaluate(x_test3, y_test3, 
                       batch_size = batch_size, 
                       verbose = 1)
print('loss3:',score3[0])
print('acc3:',score3[1])