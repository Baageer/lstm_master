from sklearn import preprocessing
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle
import random

import data_processing as dp

from LSTM_net import LSTMRNN

dp_test = dp.data_processing(['600219.SH','600170.SH','603799.SH', '600369.SH', '600372.SH',
                              '600893.SH','603288.SH','600570.SH', '601899.SH', '600837.SH',],
                         "2022-06-01", 
                         "2022-11-25", codefile="SZ50.txt")

BATCH_START = 0     # 建立 batch data 时候的 index
TIME_STEPS = 10     # backpropagation through time 的 time_steps
BATCH_SIZE = 1
INPUT_SIZE = 13     # 数据输入 size
OUTPUT_SIZE = 4     # 数据输出 size
CELL_SIZE = 64      # RNN 的 hidden unit size
LR = 0.01          # learning rate

data_col = ['open','close','high','low','pct_chg','ma5','ma20',]
input_size = len(data_col)
test_data = dp_test.data_prepare(TIME_STEPS, 1, data_col=data_col, clean_tmp=False)
testx, testy = list(zip(*test_data))

model = LSTMRNN(TIME_STEPS, input_size, OUTPUT_SIZE, CELL_SIZE, 1, LR, "train_")
saver = tf.train.Saver()
itr = 0
suma = 0
count = 0
with tf.Session() as sess:
    # 从磁盘上加载对象
    saver.restore(sess, "model/lstm_rnn.model-100")
    print("Model restored.")
    while itr < len(testx):
        ix = [testx[itr]]
        iy = [testy[itr]]
        
        if(iy[0][3] == 1):
            # print(iy)
            # print(sess.run(model.accuracy, feed_dict={model.xs: ix,model.ys: iy,}) )
            acc, pred = sess.run([model.accuracy, model.pred_softmax], feed_dict={model.xs: ix,model.ys: iy,})
            # if acc == 0.0:
            #     print(iy, pred)
            suma += acc
            count += 1
            # ttt = input()

        itr += 1

print(suma/count)