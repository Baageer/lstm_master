# 参考https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-09-RNN3/
from sklearn.datasets import load_boston
 
from sklearn import preprocessing
import tensorflow as tf
import numpy as np
import pickle
import random

from numpy import set_printoptions
set_printoptions(threshold=np.inf)

path = 'srcdata10'

f1 = open(path, 'rb')
data1 = pickle.load(f1)
#print(data1)
# 波士顿房价数据

#boston = load_boston()
# x = data1['train_x']
# y = data1['train_y']
# train_length_list = data1['train_length_list']

# x1 = data1['test_x']
# y1 = data1['test_y']
# test_length_list = data1['test_length_list']

#print(x)
#print(y)
#t = input()
 
# print('波士顿数据X:',x1.shape)# (506, 13)
# # print(x[::100])
# print('波士顿房价Y:',y1.shape)
# t = input()
# print(y[::100])
# 数据标准化
# ss_x = preprocessing.StandardScaler()
# train_x = ss_x.fit_transform(x)
# test_x = ss_x.fit_transform(x1)
# ss_y = preprocessing.StandardScaler()
# train_y = ss_y.fit_transform(y.reshape(-1, 1))
# test_y = ss_y.fit_transform(y1.reshape(-1, 1))

train_list = data1['train']
test_list = data1['test']




BATCH_START = 0     # 建立 batch data 时候的 index
TIME_STEPS = 15     # backpropagation through time 的 time_steps
BATCH_SIZE = 256
INPUT_SIZE = 13     # sin 数据输入 size
OUTPUT_SIZE = 1     # cos 数据输出 size
CELL_SIZE = 10      # RNN 的 hidden unit size
LR = 0.006          # learning rate

TEST_BATCH_START = 0
TEST_BATCH_SIZE = 1

train_idx = 0

def get_data_batch():
    global train_list, BATCH_START, TIME_STEPS
    if(BATCH_SIZE == 0):
        random.shuffle(train_list)
    
    part1 = train_list[BATCH_START : BATCH_START+BATCH_SIZE]
    
    x_part1 = []
    y_part1 = []
    for item in part1:
        
        x_part1.append(item['x'])
        y_part1.append(item['y'])    

    seq =np.array(x_part1).reshape((BATCH_SIZE, TIME_STEPS ,INPUT_SIZE))
    res =np.array(y_part1).reshape((BATCH_SIZE, 1 ,1))

    # print(seq)
    # print(res)
    # t = input()

    BATCH_START += 5

    return [seq , res]

def get_data_batch_test():
    global test_list, BATCH_START, TIME_STEPS
    
    part1 = test_list[BATCH_START : BATCH_START+TEST_BATCH_SIZE]
    
    x_part1 = []
    y_part1 = []
    for item in part1:
        
        x_part1.append(item['x'])
        y_part1.append(item['y'])

    seq =np.array(x_part1).reshape((TEST_BATCH_SIZE, TIME_STEPS ,INPUT_SIZE))
    res =np.array(y_part1).reshape((TEST_BATCH_SIZE, 1 ,1))
 
    BATCH_START += 1

    return [seq , res]
    
 
def get_batch_boston():
    global train_x, train_y,BATCH_START, TIME_STEPS
    x_part1 = train_x[BATCH_START : BATCH_START+TIME_STEPS*BATCH_SIZE]
    y_part1 = train_y[BATCH_START+TIME_STEPS : BATCH_START+TIME_STEPS*BATCH_SIZE+TIME_STEPS]
    #print('时间段=', BATCH_START, BATCH_START + TIME_STEPS * BATCH_SIZE)
 
    temp = []
    for idx, item in enumerate(y_part1):
        if idx%TIME_STEPS==0:
            temp.append(item)
    temp = np.array(temp)
    seq =x_part1.reshape((BATCH_SIZE, TIME_STEPS ,INPUT_SIZE))
    res =temp.reshape((BATCH_SIZE, 1 ,1))
 
    BATCH_START += 1
 
    # returned seq, res and xs: shape (batch, step, input)
    #np.newaxis 用来增加一个维度 变为三个维度，第三个维度将用来存上一批样本的状态
    return [seq , res  ]

def get_batch_test():
    global test_x, test_y,TEST_BATCH_START, TIME_STEPS, train_x, train_y
    x_part1 = test_x[TEST_BATCH_START : TEST_BATCH_START+TIME_STEPS*TEST_BATCH_SIZE]
    y_part1 = test_y[TEST_BATCH_START+TIME_STEPS : TEST_BATCH_START+TIME_STEPS*TEST_BATCH_SIZE+TIME_STEPS]
    print('测试时间段=', TEST_BATCH_START, TEST_BATCH_START + TIME_STEPS * TEST_BATCH_SIZE)

    temp = []
    for idx, item in enumerate(y_part1):
        if idx%TIME_STEPS==0:
            temp.append(item)
    temp = np.array(temp)
    seq =x_part1.reshape((TEST_BATCH_SIZE, TIME_STEPS ,INPUT_SIZE))
    res =temp.reshape((TEST_BATCH_SIZE, 1 ,1))
 
    TEST_BATCH_START += 1
 
    # returned seq, res and xs: shape (batch, step, input)
    #np.newaxis 用来增加一个维度 变为三个维度，第三个维度将用来存上一批样本的状态
    return [seq , res  ]
 
 
def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    print('xs.shape=',xs.shape)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # import matplotlib.pyplot as plt
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    print('增加维度前:',seq.shape)
    print(  seq[:2])
    print('增加维度后:',seq[:, :, np.newaxis].shape)
    print(seq[:2])
    # returned seq, res and xs: shape (batch, step, input)
    #np.newaxis 用来增加一个维度 变为三个维度，第三个维度将用来存上一批样本的状态
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]
 
 
class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size, name_scope):
        '''
        :param n_steps: 每批数据总包含多少时间刻度
        :param input_size: 输入数据的维度
        :param output_size: 输出数据的维度 如果是类似价格曲线的话，应该为1
        :param cell_size: cell的大小
        :param batch_size: 每批次训练数据的数量
        '''
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope(name_scope+'inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs') #xs 有三个维度
            self.ys = tf.placeholder(tf.float32, [None, 1, output_size], name='ys') #ys 有三个维度
        with tf.variable_scope(name_scope+'in_hidden'):
            self.add_input_layer()
        with tf.variable_scope(name_scope+'LSTM_cell'):
            self.add_cell()
        with tf.variable_scope(name_scope+'out_hidden'):
            self.add_output_layer()
        with tf.name_scope(name_scope+'cost'):
            self.compute_cost()
        with tf.name_scope(name_scope+'train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)
    #增加一个输入层
    def add_input_layer(self,):
        # l_in_x:(batch*n_step, in_size),相当于把这个批次的样本串到一个长度1000的时间线上，每批次50个样本，每个样本20个时刻
        l_in_x = tf.reshape(self.xs, [-1, self.input_size], name='2_2D')  #-1 表示任意行数
        # Ws (in_size, cell_size)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        # bs (cell_size, )
        bs_in = self._bias_variable([self.cell_size,])
        # l_in_y = (batch * n_steps, cell_size)
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in
        # reshape l_in_y ==> (batch, n_steps, cell_size)
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2_3D')
 
    #多时刻的状态叠加层
    def add_cell(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=0.75)
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
        #time_major=False 表示时间主线不是第一列batch
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
 
    # 增加一个输出层
    def add_output_layer(self):
        # shape = (batch * steps, cell_size)
        output_size = 10*self.n_steps
        #print(self.cell_outputs)
        l_out_x = tf.reshape(self.cell_outputs, [-1, output_size], name='2_2D')
        Ws_out = self._weight_variable([output_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out #预测结果
 
    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size ], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)
 
    def ms_error(self,labels, logits):
        return tf.square(tf.subtract(labels,logits))
 
    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)
 
    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

G_STATE = None
def train_lstm():
    
    #seq, res  = get_batch_boston()
    global BATCH_START
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE, "train_")
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs", sess.graph)
        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        sess.run(tf.global_variables_initializer())
        # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
        # $ tensorboard --logdir='logs'
        for j in range(100):#训练200次
            pred_res=None
            for i in range(800):#把整个数据分为20个时间段
                seq, res = get_data_batch()
                #print(seq)
                #print(res)
                #t = input()
                if i == 0:
                    feed_dict = {
                            model.xs: seq,
                            model.ys: res,
                            # create initial state
                    }
                else:
                    feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        model.cell_init_state: state    # use last state as the initial state for this run
                    }
    
                _, cost, state, pred = sess.run(
                    [model.train_op, model.cost, model.cell_final_state, model.pred],
                    feed_dict=feed_dict)
                pred_res=pred
                global G_STATE
                G_STATE = state
                result = sess.run(merged, feed_dict)
                writer.add_summary(result, i)
            print('{0} cost: '.format(j ), round(cost, 8))
            BATCH_START=0 #从头再来一遍
            if j % 99==0 and j != 0:
                print("j", j)
                print("保存模型：",saver.save(sess,'model/stock2.model',global_step=i))
                #print("G_STATE", G_STATE)
                np.save("state.npy:",G_STATE)

def test_lstm(test_len):
    #state = np.load("state.npy")
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, 1, "train_")
    saver=tf.train.Saver(tf.global_variables())
    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("logs", sess.graph)
        module_file = tf.train.latest_checkpoint('model/')
        saver.restore(sess, module_file)
        # relocate to the local dir and run this line to view it on Chrome (http://0.0.0.0:6006/):
        # $ tensorboard --logdir='logs'
        
        pred_res=[]
        gt_res = []
        for i in range(test_len):#把整个数据分为20个时间段
            seq, res = get_data_batch_test()


            if i == 0:
                feed_dict = {
                        model.xs: seq,
                        model.ys: res,
                        # create initial state
                }
            else:
                feed_dict = {
                    model.xs: seq,
                    model.ys: res,
                    model.cell_init_state: state    # use last state as the initial state for this run
                }

            state, pred = sess.run(
                [model.cell_final_state, model.pred],
                feed_dict=feed_dict)
            pred_res.append(pred[0])
            
            gt_res.append(res[0])

            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)
            
        return np.array(pred_res), np.array(gt_res)


def draw_prd():
    test_len = 200
    pred_test, gt_res = test_lstm(test_len)
    print(pred_test)
    
    #与最后一次训练所用的数据保持一致

    print(type(gt_res))
    print(type(gt_res.flatten()))
    print(np.abs((gt_res.flatten()-pred_test.flatten()).sum())/test_len)
 
    r_size=BATCH_SIZE * TIME_STEPS
    ###画图###########################################################################
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(20, 3))  # dpi参数指定绘图对象的分辨率，即每英寸多少个像素，缺省值为80
    axes = fig.add_subplot(1, 1, 1)
    #为了方便看，只显示了后100行数据
    show_len = 100
    indx = 0
    line1,=axes.plot(range(show_len), pred_test.flatten()[indx+1:indx+show_len+1] , 'b--',label='rnn result')
    #line2,=axes.plot(range(len(gbr_pridict)), gbr_pridict, 'r--',label='优选参数')
    line3,=axes.plot(range(show_len), gt_res.flatten()[indx:indx+show_len], 'r',label='groud truth')
 
    axes.grid()
    fig.tight_layout()
    #plt.legend(handles=[line1, line2,line3])
    plt.legend(handles=[line1,  line3])
    plt.title('RNN')
    plt.show()
 
if __name__ == '__main__':
    
    # train_lstm()

    draw_prd()
    
