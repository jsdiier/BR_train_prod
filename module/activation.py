"""
@function: 激活函数
@author: jiazhuo
"""
import tensorflow as tf
import math

class GELU(tf.keras.layers.Layer):
    """
    GELU activation (BERT style)
    Compatible with TF 2.3
    """
    def __init__(self, **kwargs):
        super(GELU, self).__init__(**kwargs)

    def call(self, inputs):
        return 0.5 * inputs * (1.0 + tf.tanh(
            math.sqrt(2.0 / math.pi) * (inputs + 0.044715 * tf.pow(inputs, 3))
        ))

    def get_config(self):
        base_config = super(GELU, self).get_config()
        return base_config


class Dice(tf.keras.layers.Layer):
    """Data-adaptive activation from DIN.

    BatchNormalization supplies moving statistics for evaluation/serving, and
    alpha is learned independently for every hidden channel.
    """
    def __init__(self, epsilon=1e-8, momentum=0.99, **kwargs):
        super(Dice, self).__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum = momentum
        self.normalizer = tf.keras.layers.BatchNormalization(
            axis=-1,
            momentum=momentum,
            epsilon=epsilon,
            center=False,
            scale=False,
        )

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(int(input_shape[-1]),),
            initializer='zeros',
            trainable=True,
        )
        super(Dice, self).build(input_shape)

    def call(self, inputs, training=None):
        probability = tf.sigmoid(self.normalizer(inputs, training=training))
        return probability * inputs + (1.0 - probability) * self.alpha * inputs

    def get_config(self):
        config = super(Dice, self).get_config()
        config.update({
            'epsilon': self.epsilon,
            'momentum': self.momentum,
        })
        return config
