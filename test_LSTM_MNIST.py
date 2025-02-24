from sklearn import preprocessing
import tensorflow.compat.v1 as tf
from keras.utils import np_utils
tf.disable_v2_behavior()
import numpy as np
import pickle
import random
from LSTM_net import LSTMRNN

tf.set_random_seed(1)   # set random seed

# hyperparameters
lr = 0.001                  # learning rate
training_iters = 100000     # train step 上限
batch_size = 128            
n_inputs = 28               # MNIST data input (img shape: 28*28)
n_steps = 28                # time steps
n_hidden_units = 128        # neurons in hidden layer
n_classes = 10              # MNIST classes (0-9 digits)




# 导入数据
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)

mnist_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
mnist_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))

mnist_train_batch = mnist_train.repeat(int(training_iters/batch_size)).batch(batch_size=batch_size)

next_op = mnist_train_batch.make_one_shot_iterator().get_next()




if __name__ == "__main__":
    model = LSTMRNN(n_steps, n_inputs, n_classes, n_hidden_units, batch_size, lr, "train_")
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        
        sess.run(init)
        step = 0
        while step * batch_size < training_iters:
            batch_xs, batch_ys = sess.run(next_op)
            try:
                batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
            except Exception as e:
                print("reshape error.", e)
                step += 1
                continue
            sess.run([model.train_op], feed_dict={
                model.xs: batch_xs,
                model.ys: batch_ys,
            })
            if step % 20 == 0:
                print(sess.run(model.accuracy, feed_dict={
                    model.xs: batch_xs,
                    model.ys: batch_ys,
                }))
            step += 1

