from sklearn import preprocessing
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import pickle
import random

import data_processing as dp

from LSTM_net import LSTMRNN

BATCH_START = 0     # 建立 batch data 时候的 index
TIME_STEPS = 10     # backpropagation through time 的 time_steps
BATCH_SIZE = 100
INPUT_SIZE = 13     # sin 数据输入 size
OUTPUT_SIZE = 1     # cos 数据输出 size
CELL_SIZE = 10      # RNN 的 hidden unit size
LR = 0.006          # learning rate

dp_train = dp.data_processing(['600219.SH','600170.SH','600219.SH', '600369.SH', '600372.SH',],
                         "2019-01-01", 
                         "2019-12-31")

dp_test = dp.data_processing(['600219.SH','600170.SH','600219.SH', '600369.SH', '600372.SH',],
                         "2020-01-01", 
                         "2020-05-31")

trainx, trainy = dp_train.data_prepare(TIME_STEPS, 1, clean_tmp=False)
testx, testy = dp_test.data_prepare(TIME_STEPS, 1, clean_tmp=False)


def get_data_batch(batch_start, batch_size, datax, datay):
    batchx = []
    batchy = []
    if batch_start+batch_size<len(trainx):
        batchx = trainx[batch_start: batch_start+batch_size]
        batchy = trainy[batch_start: batch_start+batch_size]
        print(batchy)
        tt = input()

    else :
        batchx = trainx[-batch_size:]
        batchy = trainy[-batch_size:]

    return batchx,batchy


def train_lstm(time_steps, input_size, output_size, cell_size, batch_start, batch_size, lr):
    model = LSTMRNN(time_steps, input_size, output_size, cell_size, batch_size, lr, "train_")
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)
    epoch = 0
    steps = 0
    firstflag = 0
    with tf.Session() as sess:
        # merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs", sess.graph)

        sess.run(tf.global_variables_initializer())

            
            # print(np.array(seq).shape)
        step = 0
        while epoch < 20:
            seq, res = get_data_batch(batch_start, batch_size, trainx, trainy)
            sess.run([model.train_op], feed_dict={
                model.xs: seq,
                model.ys: res,
            })
            if step % 20 == 0:
                print(sess.run(model.accuracy, feed_dict={
                    model.xs: seq,
                    model.ys: res,
                }))


            batch_start += batch_size
            steps += 1
            if batch_start >= len(trainx):
                batch_start = 0
                steps = 0
                epoch += 1

        if epoch % 100 == 0 and batch_start == 2*batch_size:
            print( '{0} cost: '.format(epoch), round(cost, 8))     
            print("保存模型： ", saver.save(sess, "model/lstm_rnn.model", global_step=epoch))



if __name__ == "__main__":
    train_lstm(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_START, BATCH_SIZE, LR)
    