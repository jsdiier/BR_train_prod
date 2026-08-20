import tensorflow as tf
from tensorflow.keras import regularizers
class DIN_attention_Layer(tf.keras.layers.Layer):
    def __init__(self, att_hidden_units, activation='relu', name=''):
        super(DIN_attention_Layer, self).__init__()
        self.name_prefix = name

        self.att_dense = [tf.keras.layers.Dense(unit, activation=activation, name=f'{name}_att_dense_{i}') for i, unit
                          in enumerate(att_hidden_units)]
        self.att_final_dense = tf.keras.layers.Dense(1, name=f'{name}_att_final_dense')
        self.const_min = -4294967295
        self.q_dense = tf.keras.layers.Dense(32, name=f'{name}_q_dense')
        self.k_dense = tf.keras.layers.Dense(32, name=f'{name}_k_dense')
        self.v_dense = tf.keras.layers.Dense(32, name=f'{name}_v_dense')

    def call(self, inputs):
        q, k, v, mask = inputs
        q = tf.expand_dims(q, axis=1)
        q = tf.tile(q, multiples=[1, tf.shape(k)[1], 1])

        q = self.q_dense(q)
        k = self.k_dense(k)
        v = self.v_dense(v)

        info = tf.concat([q, k, q - k, q * k], axis=-1)

        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)
        outputs = tf.squeeze(outputs, axis=-1)

        paddings = tf.ones_like(outputs) * self.const_min
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)

        outputs = tf.expand_dims(tf.nn.softmax(logits=outputs), axis=1)
        outputs = tf.squeeze(tf.matmul(outputs, v), axis=1)

        return outputs

class MultiHeadDINAttention(tf.keras.layers.Layer):
    def __init__(self, att_hidden_units, activation='relu', num_heads=4, head_dim=8, name=''):
        super(MultiHeadDINAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dim = num_heads * head_dim
        self.q_dense = tf.keras.layers.Dense(self.dim, name=f'{name}_mh_q_dense')
        self.k_dense = tf.keras.layers.Dense(self.dim, name=f'{name}_mh_k_dense')
        self.v_dense = tf.keras.layers.Dense(self.dim, name=f'{name}_mh_v_dense')

    def call(self, inputs):
        q, k, v, mask = inputs
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(k)[1]

        q = tf.expand_dims(q, axis=1)
        q = self.q_dense(q)
        k = self.k_dense(k)
        v = self.v_dense(v)

        def split_heads(x):
            x = tf.reshape(x, [batch_size, seq_len, self.num_heads, self.head_dim])
            return tf.transpose(x, [0, 2, 1, 3])

        qh = split_heads(q)
        kh = split_heads(k)
        vh = split_heads(v)

        scores = tf.matmul(qh, kh, transpose_b=True) / tf.sqrt(tf.cast(self.head_dim, tf.float32))
        mask_expanded = tf.expand_dims(tf.expand_dims(mask, axis=1), axis=1)
        scores = tf.where(tf.equal(mask_expanded, 0), -1e9, scores)
        att = tf.nn.softmax(scores, axis=-1)

        out = tf.reduce_sum(att * vh, axis=-2)
        out = tf.reshape(out, [batch_size, self.num_heads * self.head_dim])
        return out
