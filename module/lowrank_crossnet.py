import tensorflow as tf


class LowRankCrossLayer(tf.keras.layers.Layer):
    """A DCN-V2 cross layer with W represented by U @ V."""

    def __init__(self, low_rank=64, **kwargs):
        super().__init__(**kwargs)
        self.low_rank = int(low_rank)
        self.down = tf.keras.layers.Dense(
            self.low_rank, use_bias=False, name="down")
        self.up = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        if input_dim is None:
            raise ValueError("LowRankCrossLayer requires a static input dimension")
        self.up = tf.keras.layers.Dense(
            int(input_dim), use_bias=True, name="up")
        super().build(input_shape)

    def call(self, inputs):
        x0, x = inputs
        cross = self.up(self.down(x))
        return x + x0 * cross


class LowRankCrossNet(tf.keras.layers.Layer):
    """Stacked low-rank DCN-V2 layers."""

    def __init__(self, num_layers=2, low_rank=64, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = int(num_layers)
        self.low_rank = int(low_rank)
        self.cross_layers = [
            LowRankCrossLayer(low_rank=self.low_rank, name="cross_%d" % i)
            for i in range(self.num_layers)
        ]

    def call(self, x0):
        x = x0
        for layer in self.cross_layers:
            x = layer([x0, x])
        return x
