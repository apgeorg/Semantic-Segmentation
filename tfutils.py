import tensorflow as tf

# custom init with the seed set to 0 by default
def custom_init(stddev):
    return tf.random_normal_initializer(stddev=stddev)

def l2_regularizer(l2_weight):
    return tf.contrib.layers.l2_regularizer(l2_weight)

def conv_1x1(x, num_outputs, name):
    return tf.layers.conv2d(x, num_outputs, kernel_size=1, padding='same',
                            kernel_initializer=custom_init(stddev=0.01),
                            kernel_regularizer=l2_regularizer(1e-3),
                            name=name)

def upsample(x, num_outputs, kernel_size, stride, name):
    return tf.layers.conv2d_transpose(x, num_outputs, kernel_size=kernel_size, strides=stride, padding='same',
                                      kernel_initializer=custom_init(stddev=0.01),
                                      kernel_regularizer=l2_regularizer(1e-3),
                                      name=name)
