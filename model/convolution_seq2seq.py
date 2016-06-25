import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import rnn
from tensorflow.python.ops.seq2seq import rnn_decoder


def init_W(shape):
    init = tf.truncated_normal(shape)
    return tf.Variable(init)


def init_bias(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1], padding='SAME')


def maxpooling_2x2(image):
    return tf.nn.max_pool(image, ksize=[1, 2, 2, 1],
                                      stride=[1, 2, 2, 1], padding='SAME')


def deconv2d(x, W):
    return tf.nn.conv2d_transpose(input=x, filter=W, strides=[1, 1, 1, 1],
                                  padding='SAME')


class ConvolutionSeq2Seq(object):
    def __init__(self, inputs, outputs,
                 seq_length, n_hidden=100, lstm_hidden=10,
                 kernels=[100, 100, 100]):
        self.n_hidden = n_hidden
        self.lstm_hidden = lstm_hidden
        self.inputs = inputs
        self.outputs = outputs
        self.seq_length = seq_length

    def prepare_graph(self):
        # prepare the input batchholder
        with tf.name_scope("encoder/decoder/convNet"):
            # declare the place holder for a batch of seq
            inputs = []
            targets = []
            # input and targes are both images
            for i in xrange(self.seq_length):
                inputs += [tf.placeholder(dtype=tf.float32,
                                          shape=(None, self.image_shape[0],
                                                 self.image_shape[1]))]
                targets += [tf.placeholder(dtype=tf.float32,
                                           shape=(None, self.image_shape[0],
                                                  self.image_shape[1]))]
            # initial the weights and bias for endcoder and decoder
            # fot each convolution kernel shared the same weights
            self.W_conv = init_W(shape=[20, 20, 1, 32])
            self.b_conv = init_bias(shape=[32])
            encoder_conv = []
            encoder_max = []

            for input, target in zip(inputs, targets):
                encoder_conv += [tf.nn.relu(conv2d(input, self.W_conv) +
                                            self.b_conv)]
                encoder_max += [maxpooling_2x2(encoder_conv[-1])]

        with variable_scope.variable_scope("LSTM-CovolutionSeq2Seq"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.lstm_hidden)
            _, enc_state = rnn.rnn(cell, encoder_conv, dtype=tf.float32)
            # put enc_states  into a convolutional net

            decoders, state = rnn_decoder(encoder_max, enc_state, cell,
                                          feed_previous=True)

            for decoder in decoders:  # upsampling
                conv2d(decoder, tf.transpose(self.W_conv, premu=[2, 3, 0, 1]))

            # decoder

    def step(self):
        pass
