from keras.optimizers import Optimizer
import keras.backend as K
import tensorflow as tf


def unitwise_norm(x):
    if len(x.shape) == 1:  # bias (out,)
        # axis = None
        # keepdims = True
        return x
    elif len(x.shape) == 2:  # dense weight (in,out)
        axis = 0
        keepdims = True
    elif len(x.shape) == 4:  # kernel (k,k,in,out)
        axis = [0, 1, 2,]
        keepdims = True
    else:
        raise ValueError(f'Got a parameter with shape not in [1, 2, 4]! {x}')
    return K.sqrt(K.sum(x*x, axis=axis, keepdims=keepdims))


class AGCSGD(Optimizer):
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False,
                 clipping=1e-2, eps=1e-3, level_groups=None, **kwargs):
        super(AGCSGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.clipping = clipping
        self.eps = eps
        self.level_groups = level_groups

    def get_updates(self, loss, params):

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * tf.cast(self.iterations, K.dtype(self.decay))))

        grads = self.get_gradients(loss, params)    # [N,M]
        self.updates = [K.update_add(self.iterations, 1)]   # values to update by each step
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]   # [N,M]
        self.weights = [self.iterations] + moments    # values to continue training when reloaded

        for p, g, m in zip(params, grads, moments):
            # clip g
            p_norm = unitwise_norm(p)
            g_norm = unitwise_norm(g)
            max_norm = K.maximum(p_norm*self.clipping, self.eps)
            cond = g_norm > max_norm
            g = K.switch(cond, lambda:g / g_norm * max_norm, lambda:g)
            # update p
            v = self.momentum * m - lr * g
            self.updates.append(K.update(m, v))     # update velocity
            if self.nesterov:
                new_p = p + self.momentum * v - lr * g
            else:
                new_p = p + v
            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))
            return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'clipping': self.clipping,
                  'eps': self.eps,
                  'level_groups': self.level_groups,
                  }
        base_config = super(AGCSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


if __name__ == '__main__':

    import keras
    import numpy as np
    # Construct and compile an instance of CustomModel
    inputs = keras.Input(shape=(32,32,3))
    outputs = keras.layers.Conv2D(16, 3, strides=(1, 1), padding='same')(inputs)
    outputs = keras.layers.GlobalAveragePooling2D()(outputs)
    outputs = keras.layers.Dense(10)(outputs)
    model = keras.Model(inputs, outputs)

    def get_layer_group(model):
        level_groups = {}
        for layer in model.layers:   # exclude the linear head
            names = [i.name for i in layer.weights]
            if names:
                level_groups[layer.name] = names
        return level_groups
    level_groups = get_layer_group(model)
    print(level_groups)

    model.compile(AGCSGD(lr=1.6, level_groups=level_groups), loss="mse", metrics=["mae"])

    # Just use `fit` as usual
    x = np.random.random((1000, 32, 32, 3))
    y = np.random.random((1000, 10))
    model.fit(x, y, epochs=30, batch_size=100)



