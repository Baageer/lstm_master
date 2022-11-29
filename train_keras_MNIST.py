import keras
from keras import Sequential
from keras.layers import LSTM,Dense, Activation
from keras import optimizers
from keras.datasets import mnist
#加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train.shape:',x_train.shape)
print('x_test.shape:',x_test.shape)

#时间序列数量
n_step = 28
#每次输入的维度
n_input = 28
#分类类别数
n_classes = 10

#将数据转为28*28的数据（n_samples,height,width）
x_train = x_train.reshape(-1, n_step, n_input)
x_test = x_test.reshape(-1, n_step, n_input)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#标准化数据，因为像素值在0-255之间，所以除以255后数值在0-1之间
x_train /= 255
x_test /= 255

#y_train，y_test 进行 one-hot-encoding，label为0-9的数字，所以一共10类
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)



#学习率
learning_rate = 0.001
#每次处理的数量
batch_size = 28
#循环次数
epochs = 20
#神经元的数量
n_lstm_out = 128

model = Sequential()

#LSTM层
model.add(LSTM(
        units = n_lstm_out,
        input_shape = (n_step, n_input)))
#全连接层          
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

#训练
model.fit(x_train, y_train, 
          epochs = epochs, 
          batch_size= batch_size, 
          verbose = 1, 
          validation_data = (x_test,y_test))
#评估
score = model.evaluate(x_test, y_test, 
                       batch_size = batch_size, 
                       verbose = 1)
print('loss:',score[0])
print('acc:',score[1])
