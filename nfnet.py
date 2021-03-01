from keras.layers import Lambda, Input, AveragePooling2D, GlobalAveragePooling2D, multiply, Dropout, Dense, add, Activation
from keras.initializers import RandomNormal
from keras.models import Model
import tensorflow as tf
from WSConv2D import WSConv2D
from utils import nfnet_params


def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return x*cdf


def NFNet(n_classes=10, variant='F0', se_ratio=0.5, alpha=0.2, stochdepth_rate=0.1, activation=gelu,
          final_conv_mult=2, training=True):
    # configurations
    block_params = nfnet_params[variant]
    width_pattern = block_params['width']
    depth_pattern = block_params['depth']
    imsize = block_params['train_imsize'] if training else block_params['test_imsize']
    bneck_pattern = block_params.get('expansion', [0.5] * 4)
    group_pattern = block_params.get('group_width', [128] * 4)
    drop_rate = block_params['drop_rate']
    stride_pattern = [1, 2, 2, 2]

    # stem
    inpt = Input((imsize, imsize, 3))
    x = WSConv2D(16, 3, strides=2, padding='same')(inpt)
    x = Activation(activation)(x)
    x = WSConv2D(32, 3, strides=1, padding='same')(x)
    x = Activation(activation)(x)
    x = WSConv2D(64, 3, strides=1, padding='same')(x)
    x = Activation(activation)(x)
    x = WSConv2D(width_pattern[0]//2, 3, strides=2, padding='same', groups=1)(x)

    # blocks
    num_blocks = sum(depth_pattern)
    block_idx = 0
    expected_std = 1.0
    for i in range(4):
        for j in range(depth_pattern[i]):
            strides = stride_pattern[i] if j==0 else 1
            block_stochdepth_rate = stochdepth_rate * block_idx / num_blocks
            beta = 1. / expected_std
            x = NFBlock(x, width_pattern[i], strides, activation, bneck_pattern[i],
                        group_pattern[i], alpha, beta, block_stochdepth_rate, training)
            block_idx += 1
            if j==0:
                # variance reset after each transition block
                expected_std = 1.0
            expected_std = (expected_std ** 2 + alpha ** 2) ** 0.5

    # final conv
    x = WSConv2D(width_pattern[-1]*final_conv_mult, 1, strides=1, padding='same')(x)
    x = Activation(activation)(x)

    # head
    x = GlobalAveragePooling2D()(x)
    if drop_rate and training:
        x = Dropout(drop_rate)(x)
    x = Dense(n_classes, kernel_initializer=RandomNormal(mean=0, stddev=0.01), use_bias=True)(x)

    model = Model(inpt, x)

    return model


def NFBlock(inpt, filters, strides, activation, bneck_ratio, group_width,
            alpha, beta, drop_rate, training, se_ratio=0.5):
    n_filters = int(filters*bneck_ratio)
    groups = n_filters//group_width

    # beta downscale
    inpt1 = Activation(activation)(Lambda(lambda x: x*beta)(inpt))
    # bottleneck: 1x1 + 3x3gc + 3x3gc + 1x1
    x = WSConv2D(n_filters, 1, strides=1, padding='same')(inpt1)
    x = Activation(activation)(x)
    x = WSConv2D(n_filters, 3, strides=strides, groups=groups, padding='same')(x)
    x = Activation(activation)(x)
    x = WSConv2D(n_filters, 3, strides=1, groups=groups, padding='same')(x)
    x = Activation(activation)(x)
    x = WSConv2D(filters, 1, strides=1, padding='same')(x)
    # se-block
    x = SEBlock(x, se_ratio)
    # alpha rescale
    x = Lambda(lambda x: x*alpha)(x)
    # batch-wise dropout
    x = StochasticDepth(x, drop_rate, training)

    # skip connection
    if strides>1 or inpt.shape[-1]!=filters:
        # transition block: skip after downscale
        if strides>1:
            # avg pooling
            skip = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(inpt1)
        else:
            skip = inpt1
        # 1x1 conv
        skip = WSConv2D(filters, 1, padding='same')(skip)
    else:
        # non-transistion block: skip before downscale
        skip = inpt
    # add fuse
    return add([x, skip])


def SEBlock(inpt, se_ratio, rescale_factor=2):
    filters = int(inpt.shape[-1])
    n_filters = int(filters*se_ratio)
    x = GlobalAveragePooling2D()(inpt)
    x = Dense(n_filters, activation='relu', use_bias=True)(x)
    x = Dense(filters, activation='sigmoid', use_bias=True)(x)
    # rescale
    x = Lambda(lambda x: x*rescale_factor)(x)
    return multiply([inpt, x])


def StochasticDepth(x, drop_rate, training, scale_by_keep=False):
    # Batchwise Dropout used in EfficientNet, optionally sans rescaling
    if not training:
        return x
    if drop_rate>0:
        x = Dropout(drop_rate, noise_shape=(None, 1, 1, 1))(x)
    if scale_by_keep:
        keep_prob = 1 - drop_rate
        x = Lambda(lambda x: x/keep_prob)(x)
    return x


if __name__ == '__main__':

    model = NFNet(n_classes=10, variant='F0')
    model.summary()
    model.save_weights("nfnet_f0.h5")
















