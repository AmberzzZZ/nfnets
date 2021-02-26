import tensorflow as tf
from groupConv import GroupConv2DKT
from keras.models import Model


class WSConv2D(GroupConv2DKT):

    def __init__(self, filters, kernel_size, groups=1, *args, **kwargs):
        super(WSConv2D, self).__init__(filters=filters,
                                       kernel_size=kernel_size,
                                       groups=groups,
                                       kernel_initializer="he_normal",
                                       *args, **kwargs)

    def call(self, input):
        # standardize kernel
        self.kernel = self.standardize_weight(self.kernel)
        # run conv
        if self.groups==1:
            # normal conv
            x = tf.nn.conv2d(input, self.kernel, self.strides,
                             padding=self.padding_pattern[self.padding],
                             data_format=self.dim_pattern[self.data_format])
            if self.bias:
                x = tf.nn.bias_add(x, self.bias, data_format=self.dim_pattern[self.data_format])
            if self.activation:
                return self.activation(x)
        else:
            # group conv
            return super().call(input)

    def standardize_weight(self, weight, eps=1e-4):
        # weight: (k,k,in,out)
        mean = tf.math.reduce_mean(weight, axis=(0, 1, 2), keepdims=True)   # [N,k]
        var = tf.math.reduce_variance(weight, axis=(0, 1, 2), keepdims=True)  # [N,k]
        fan_in = tf.reduce_prod(weight, axis=(0, 1, 2))   # N
        gain = self.add_weight(name='gain',
                               shape=(weight.shape[-1],),
                               initializer="ones",
                               trainable=True)
        scale = tf.math.rsqrt(tf.math.maximum(var*fan_in, tf.constant(eps, dtype=tf.float32))) * gain
        return weight * scale - (mean * scale)


if __name__ == '__main__':

    from keras.layers import Input

    x = Input((192,192,3))

    y = WSConv2D(64, 7, strides=2, padding='same', use_bias=True, groups=1)(x)
    model = Model(x,y)
    model.summary()

    y = WSConv2D(64, 7, strides=2, padding='same', use_bias=True, groups=4)(x)
    model = Model(x,y)
    model.summary()
