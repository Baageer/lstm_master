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
BATCH_SIZE = 64
INPUT_SIZE = 13     # 数据输入 size
OUTPUT_SIZE = 4     # 数据输出 size
CELL_SIZE = 64      # RNN 的 hidden unit size
LR = 0.01          # learning rate

dp_train = dp.data_processing(['600219.SH','600170.SH','603799.SH', '600369.SH', '600372.SH',
                               '600893.SH','603288.SH','600570.SH', '601899.SH', '600837.SH',],
                         "2016-01-01", 
                         "2021-12-31", codefile="SZ50.txt")

dp_test = dp.data_processing(['600219.SH','600170.SH','603799.SH', '600369.SH', '600372.SH',
                              '600893.SH','603288.SH','600570.SH', '601899.SH', '600837.SH',],
                         "2022-01-01", 
                         "2022-05-31", codefile="SZ50.txt")




def get_data_batch(batch_start, batch_size, datax, datay):
    batchx = []
    batchy = []
    if batch_start+batch_size<len(datax):
        batchx = datax[batch_start: batch_start+batch_size]
        batchy = datay[batch_start: batch_start+batch_size]


    else :
        batchx = datax[-batch_size:]
        batchy = datay[-batch_size:]

    return batchx,batchy


def train_lstm(time_steps, input_size, output_size, cell_size, batch_start, batch_size, lr):

    data_col = ['open','close','high','low','pct_chg','ma5','ma20',]
    input_size = len(data_col)
    train_data = dp_train.data_prepare(TIME_STEPS, 1, data_col=data_col, clean_tmp=False)
    train_data = dp_train.split_dataclass()
    test_data = dp_test.data_prepare(TIME_STEPS, 1, data_col=data_col, clean_tmp=False)

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
        while epoch < 101:
            trainx, trainy = list(zip(*train_data))

            seq, res = get_data_batch(batch_start, batch_size, trainx, trainy)
            # for i in seq:
            #     if(len(i)==1):
            #         print("seq" , i)
            #         t = input()
            # for i in res:
            #     if(len(i)==1):
            #         print("res" , i)
            #         t = input()
            sess.run([model.train_op], feed_dict={
                model.xs: seq,
                model.ys: res,
            })
            if steps % 300 == 0:
                testx, testy = list(zip(*test_data))
                sum = 0
                itr = int(len(testx)/batch_size-1)
                for i in range(itr):
                    seq_test, res_test = get_data_batch(i*batch_size, batch_size, testx, testy)
                    acc = sess.run(model.accuracy, feed_dict={model.xs: seq_test,model.ys: res_test,})
                    # print("test acc: ", acc)
                    sum += acc
                test_accurncy = sum/itr
                print("epoch:", epoch, "\t|  steps:", steps,
                      "\t|  train accuracy:", sess.run(model.accuracy, feed_dict={model.xs: seq,model.ys: res,}),
                      "\t|  test  accuracy:", test_accurncy,
                )

            # if epoch == 5:
            #     seq_test, res_test = get_data_batch(0, batch_size, testx, testy)
            #     print("model prediction result:", sess.run(model.pred, feed_dict={model.xs: seq_test,model.ys: res_test,}))
            #     t = input()


            batch_start += batch_size
            steps += 1
            if batch_start >= len(trainx):
                batch_start = 0
                steps = 0
                epoch += 1
                train_data = dp_train.split_dataclass()
                # test_data = dp_test.split_dataclass()
                
                # random.shuffle(train_data)

            if epoch % 100 == 0 and epoch > 0:   
                print("保存模型： ", saver.save(sess, "model/lstm_rnn.model", global_step=epoch))



if __name__ == "__main__":
    train_lstm(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_START, BATCH_SIZE, LR)
    