import tensorflow as tf
import numpy as np

data = np.random.random(size=(100, 30,  20, 20))

sess = tf.Session()
x = tf.placeholder(dtype=tf.float32, shape=(100, 30, 20, 20))
image = tf.reshape(x, shape=[100, 30, 20, 20])
init = tf.truncated_normal(shape=(5, 5, 20, 10))
W = tf.Variable(dtype=tf.float32, initial_value=init)
output = tf.nn.conv2d(image, W, strides=[1, 2, 2, 1], padding="SAME",
                      use_cudnn_on_gpu=True)

output, argmax = tf.nn.max_pool_with_argmax(output, ksize=[1, 4, 4, 1],
                                            strides=[1, 1, 1, 1],
                                            padding='SAME')
outputs = tf.nn.conv2d_transpose(output, W,
                                 [100, 30, 20, 20], strides=[1, 2, 2, 1],
                                 padding='SAME')
# Before starting, initialize the variables.  We will 'run' this first.
# Launch the graph.
sess.run(tf.initialize_all_variables())
# tf.nn.conv2d()
print outputs.eval(feed_dict={x: data}, session=sess).shape
# upsampling


# name_scope test
with tf.variable_scope("foo"):
    y = tf.Variable(dtype=tf.float32, initial_value=init, name='x')
    with tf.name_scope("test"):
        x = tf.get_variable(dtype=tf.float32, shape=(10,), name='x')
        b = x + 1
