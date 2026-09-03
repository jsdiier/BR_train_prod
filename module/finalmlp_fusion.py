import tensorflow as tf


class StreamFeatureGate(tf.keras.layers.Layer):
    """Feature gate on a compact stream representation."""

    def __init__(self, stream_dim=256, gate_hidden_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name='norm')
        self.projection = tf.keras.layers.Dense(
            stream_dim, activation=tf.nn.swish, name='projection')
        self.gate_hidden = tf.keras.layers.Dense(
            gate_hidden_dim, activation=tf.nn.swish, name='gate_hidden')
        self.gate_output = tf.keras.layers.Dense(
            stream_dim, activation='sigmoid',
            kernel_initializer='zeros', bias_initializer='zeros',
            name='gate_output')
        self.last_gate_mean = tf.constant(1.0, dtype=tf.float32)
        self.last_gate_saturation = tf.constant(0.0, dtype=tf.float32)

    def call(self, inputs):
        projected = self.projection(self.norm(inputs))
        # 2*sigmoid(0)=1, so the gate starts as an identity multiplier.
        gate = 2.0 * self.gate_output(self.gate_hidden(projected))
        self.last_gate_mean = tf.reduce_mean(gate)
        self.last_gate_saturation = tf.reduce_mean(tf.cast(
            tf.logical_or(gate < 0.1, gate > 1.9), tf.float32))
        return projected * gate


class FinalMLPTwoStreamFusion(tf.keras.layers.Layer):
    """A compact FinalMLP-inspired user/item two-stream side branch."""

    def __init__(self, stream_dim=256, gate_hidden_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.user_stream = StreamFeatureGate(
            stream_dim, gate_hidden_dim, name='user_stream')
        self.item_stream = StreamFeatureGate(
            stream_dim, gate_hidden_dim, name='item_stream')
        self.interaction = tf.keras.layers.Dense(
            stream_dim, activation=tf.nn.swish, name='interaction_aggregation')

    def call(self, inputs):
        user_inputs, item_inputs = inputs
        user = self.user_stream(user_inputs)
        item = self.item_stream(item_inputs)
        interaction_inputs = tf.concat(
            [user, item, user * item, tf.abs(user - item)], axis=-1)
        return self.interaction(interaction_inputs)

    def diagnostics(self):
        return {
            'user_gate_mean': self.user_stream.last_gate_mean,
            'user_gate_saturation': self.user_stream.last_gate_saturation,
            'item_gate_mean': self.item_stream.last_gate_mean,
            'item_gate_saturation': self.item_stream.last_gate_saturation,
        }
