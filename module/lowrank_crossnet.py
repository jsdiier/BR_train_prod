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
        # ``call`` receives ``[x0, x]``, so Keras passes a pair of
        # TensorShapes here rather than one TensorShape.
        if not isinstance(input_shape, (list, tuple)) or len(input_shape) != 2:
            raise ValueError(
                "LowRankCrossLayer expects [x0, x] input shapes, got %r"
                % (input_shape,))
        x0_shape = tf.TensorShape(input_shape[0])
        x_shape = tf.TensorShape(input_shape[1])
        input_dim = x_shape[-1]
        if input_dim is None:
            raise ValueError("LowRankCrossLayer requires a static input dimension")
        if x0_shape[-1] != input_dim:
            raise ValueError(
                "LowRankCrossLayer requires x0 and x to have the same last "
                "dimension, got %s and %s" % (x0_shape[-1], input_dim))
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
