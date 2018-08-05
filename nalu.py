
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.layers import *
from keras.models import *

class NALU(Layer):
    def __init__(self, units, kernel_initializer='glorot_uniform',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(NALU, self).__init__(**kwargs)
        self.units = units
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.W_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='W_hat')
        self.M_hat = self.add_weight(shape=(input_dim, self.units),
                                     initializer=self.kernel_initializer,
                                     name='M_hat')
        self.G = self.add_weight(shape=(input_dim, self.units),
                                 initializer=self.kernel_initializer,
                                 name='G')
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        W = K.tanh(self.W_hat) * K.sigmoid(self.M_hat) #NAC
        m = K.exp(K.dot(K.log(K.abs(inputs) + 1e-7), W)) #NALU
        g = K.sigmoid(K.dot(inputs, self.G)) #NALU
        a = K.dot(inputs, W) #NAC
        y = g * a + (1 - g) * m #NALU
        return y

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)

    def get_config(self):
        config = {
            'units': self.units,
            'kernel_initializer': initializers.serialize(self.kernel_initializer)
        }
        base_config = super(Dense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

if __name__ == "__main__":
    x = Input((10,))
    y = NALU(1)(x)
    m = Model(x, y)
    m.compile("adam", "mse")
    m.fit(np.random.rand(128, 10), np.random.rand(128, 1),batch_size=32, epochs=2000, verbose=1)
