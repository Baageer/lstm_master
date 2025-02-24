import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout, GRU, Conv1D, ReLU, BatchNormalization, Flatten
from keras import optimizers
from sklearn.utils import class_weight
import numpy as np

import data_processing as dp

#加载数据
dp_train = dp.data_processing(['600219.SH','600170.SH','603799.SH', '600369.SH', '600372.SH',
                               '600893.SH','603288.SH','600570.SH', '601899.SH', '600837.SH',],
                         "2016-01-01", 
                         "2021-12-31",codefile="SZ50.txt")

dp_test = dp.data_processing(['600219.SH','600170.SH','603799.SH', '600369.SH', '600372.SH',
                              '600893.SH','603288.SH','600570.SH', '601899.SH', '600837.SH',],
                         "2022-01-01", 
                         "2022-05-31",codefile="SZ50.txt")




# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print('x_train.shape:',x_train.shape)
# print('x_test.shape:',x_test.shape)

#时间序列数量
n_step = 30
#每次输入的维度
n_input = 10
#分类类别数
n_classes = 4

#学习率
learning_rate = 0.001
#每次处理的数量
batch_size = 128
#循环次数
epochs = 200
#神经元的数量
n_lstm_out = 128

# #将数据转为28*28的数据（n_samples,height,width）
# x_train = x_train.reshape(-1, n_step, n_input)
# x_test = x_test.reshape(-1, n_step, n_input)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# #标准化数据，因为像素值在0-255之间，所以除以255后数值在0-1之间
# x_train /= 255
# x_test /= 255

# #y_train，y_test 进行 one-hot-encoding，label为0-9的数字，所以一共10类
# y_train = keras.utils.to_categorical(y_train, n_classes)
# y_test = keras.utils.to_categorical(y_test, n_classes)

data_col = ['open','close','high','low','pct_chg','ma5','ma20', 'ma50', 'vol', 'amount']
input_size = len(data_col)
train_data = dp_train.data_prepare(n_step, 1, data_col=data_col, clean_tmp=False)
train_data = dp_train.split_dataclass()
test_data = dp_test.data_prepare(n_step, 1, data_col=data_col, clean_tmp=False)

# test_data3 = []
# for data in test_data:
#     if data[1][3] == 1:
#         test_data3.append(data)

x_train, y_train = list(zip(*train_data))
x_test, y_test = list(zip(*test_data))
x_train, y_train = np.array(x_train), np.array(y_train)
x_test, y_test = np.array(x_test), np.array(y_test)


# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape,)
# ttt = input()


model = Sequential()

model.add(Conv1D(64, 3, strides=1,input_shape=(n_step, n_input), use_bias=False))
model.add(ReLU())

model.add(Conv1D(64, 3, strides=1))  # [None, 54, 64]
model.add(BatchNormalization())

model.add(LSTM(64, dropout=0.5, return_sequences=True))
model.add(LSTM(64, dropout=0.5, return_sequences=True))
model.add(LSTM(64))
#全连接层          
model.add(Dense(units = 128))
model.add(Dense(units = n_classes))
#激活层
model.add(Activation('softmax'))

#查看各层的基本信息
model.summary()

# 编译
model.compile(
    optimizer = optimizers.Adam(lr = learning_rate),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy'])

# from keras.models import load_model
# model = load_model('model-500.h5')

#训练
model.fit(x_train, y_train, 
          epochs = epochs, 
          batch_size= batch_size, 
          verbose = 1, 
          validation_data = (x_test,y_test),
          class_weight = 'auto')
#评估
score = model.evaluate(x_test, y_test, 
                       batch_size = batch_size, 
                       verbose = 1)
print('loss:',score[0])
print('acc:',score[1])

model.save("CNN_LSTM_model-200.h5")