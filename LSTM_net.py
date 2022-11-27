import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
# from tensorflow_core import seq_loss

class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size,lr, name_scope):
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
            self.ys = tf.placeholder(tf.float32, [None, output_size], name='ys') #ys 有三个维度
        with tf.variable_scope(name_scope+'in_hidden'):
            self.add_input_layer()
        with tf.variable_scope(name_scope+'LSTM_cell'):
            self.add_cell()
        with tf.variable_scope(name_scope+'out_hidden'):
            self.add_output_layer()
        with tf.name_scope(name_scope+'cost'):
            self.compute_cost()
        with tf.name_scope(name_scope+'train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.cost)
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
        # output_size = 10*self.n_steps
        #print(self.cell_outputs)
        # l_out_x = tf.reshape(self.cell_outputs, [-1, output_size], name='2_2D')
        l_out_x = tf.unstack(tf.transpose(self.cell_outputs, [1, 0, 2]))

        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size, ])
        # shape = (batch * steps, output_size)
        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x[-1], Ws_out) + bs_out #预测结果

        self.pred_softmax = tf.nn.softmax(self.pred)

        self.correct_pred = tf.equal(tf.argmax(tf.nn.softmax(self.pred), 1), tf.argmax(self.ys, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
 
    def compute_cost(self):
        # losses = seq_loss.sequence_loss_by_example(
        #     [tf.reshape(self.pred, [-1], name='reshape_pred')],
        #     [tf.reshape(self.ys, [-1], name='reshape_target')],
        #     [tf.ones([self.batch_size ], dtype=tf.float32)],
        #     average_across_timesteps=True,
        #     softmax_loss_function=self.ms_error,
        #     name='losses'
        # )
        # with tf.name_scope('average_cost'):
        #     self.cost = tf.div(
        #         tf.reduce_sum(losses, name='losses_sum'),
        #         self.batch_size,
        #         name='average_cost')
        #     tf.summary.scalar('cost', self.cost)
        with tf.name_scope('average_cost'):
            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.ys))
 
    def ms_error(self,labels, logits):
        return tf.square(tf.subtract(labels,logits))
 
    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)
 
    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)