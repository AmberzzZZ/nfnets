from keras.activations import elu, relu, selu, sigmoid, softplus, tanh
from keras.layers import Lambda
import tensorflow as tf


## nfnet family
nfnet_params = {
    'F0': {
        'width': [256, 512, 1536, 1536], 'depth': [1, 2, 6, 3],
        'train_imsize': 192, 'test_imsize': 256,
        'RA_level': '405', 'drop_rate': 0.2},
    'F1': {
        'width': [256, 512, 1536, 1536], 'depth': [2, 4, 12, 6],
        'train_imsize': 224, 'test_imsize': 320,
        'RA_level': '410', 'drop_rate': 0.3},
    'F2': {
        'width': [256, 512, 1536, 1536], 'depth': [3, 6, 18, 9],
        'train_imsize': 256, 'test_imsize': 352,
        'RA_level': '410', 'drop_rate': 0.4},
    'F3': {
        'width': [256, 512, 1536, 1536], 'depth': [4, 8, 24, 12],
        'train_imsize': 320, 'test_imsize': 416,
        'RA_level': '415', 'drop_rate': 0.4},
    'F4': {
        'width': [256, 512, 1536, 1536], 'depth': [5, 10, 30, 15],
        'train_imsize': 384, 'test_imsize': 512,
        'RA_level': '415', 'drop_rate': 0.5},
    'F5': {
        'width': [256, 512, 1536, 1536], 'depth': [6, 12, 36, 18],
        'train_imsize': 416, 'test_imsize': 544,
        'RA_level': '415', 'drop_rate': 0.5},
    'F6': {
        'width': [256, 512, 1536, 1536], 'depth': [7, 14, 42, 21],
        'train_imsize': 448, 'test_imsize': 576,
        'RA_level': '415', 'drop_rate': 0.5},
    'F7': {
        'width': [256, 512, 1536, 1536], 'depth': [8, 16, 48, 24],
        'train_imsize': 480, 'test_imsize': 608,
        'RA_level': '415', 'drop_rate': 0.5},
}

nfnet_params.update(**{ **{f'{key}+': {**nfnet_params[key], 'width': [384, 768, 2048, 2048],} for key in nfnet_params} })


## nonlinearities: add magic constants
def gelu(x):
    cdf = 0.5 * (1.0 + tf.erf(x / tf.sqrt(2.0)))
    return x*cdf

nonlinearities = {
    'identity': Lambda(lambda x: x),
    'celu': Lambda(lambda x: tf.nn.crelu(x) * 1.270926833152771),
    'elu': Lambda(lambda x: elu(x) * 1.2716004848480225),
    'gelu': Lambda(lambda x: gelu(x) * 1.7015043497085571),
#     'glu': lambda x: jax.nn.glu(x) * 1.8484294414520264,
    'leaky_relu': Lambda(lambda x: tf.nn.leaky_relu(x) * 1.70590341091156),
    'log_sigmoid': Lambda(lambda x: tf.math.log(tf.nn.sigmoid(x)) * 1.9193484783172607),
    'log_softmax': Lambda(lambda x: tf.math.log(tf.nn.softmax(x)) * 1.0002083778381348),
    'relu': Lambda(lambda x: relu(x) * 1.7139588594436646),
    'relu6': Lambda(lambda x: tf.nn.relu6(x) * 1.7131484746932983),
    'selu': Lambda(lambda x: selu(x) * 1.0008515119552612),
    'sigmoid': Lambda(lambda x: sigmoid(x) * 4.803835391998291),
    'silu': Lambda(lambda x: tf.nn.silu(x) * 1.7881293296813965),
    'soft_sign': Lambda(lambda x: tf.nn.softsign(x) * 2.338853120803833),
    'softplus': Lambda(lambda x: softplus(x) * 1.9203323125839233),
    'tanh': Lambda(lambda x: tanh(x) * 1.5939117670059204),
}


