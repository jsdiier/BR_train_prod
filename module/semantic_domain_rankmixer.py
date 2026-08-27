import tensorflow as tf

from logger import logger


class HeadedTokenAxisMixing(tf.keras.layers.Layer):
    """Head-specific learned residual mixing across semantic domain tokens."""

    def __init__(self, token_count, token_dim, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_count = token_count
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.head_mixers = [
            tf.keras.layers.Dense(
                token_count,
                use_bias=False,
                kernel_initializer='zeros',
                name='token_axis_head_%d' % head_index)
            for head_index in range(num_heads)
        ]
        self.last_residual_ratio = tf.constant(0.0, dtype=tf.float32)
        self.last_offdiag_ratio = tf.constant(0.0, dtype=tf.float32)

    def call(self, x, token_mask):
        batch_size = tf.shape(x)[0]
        mask_3d = tf.expand_dims(tf.cast(token_mask, x.dtype), -1)
        masked_x = x * mask_3d
        headed = tf.reshape(
            masked_x,
            [batch_size, self.token_count, self.num_heads, self.head_dim])
        headed = tf.transpose(headed, [0, 2, 3, 1])  # [B, H, Dh, T]

        mixed_heads = []
        for head_index, mixer in enumerate(self.head_mixers):
            mixed_heads.append(mixer(headed[:, head_index, :, :]))
        learned_delta = tf.stack(mixed_heads, axis=1)  # [B, H, Dh, T]
        learned_delta = tf.transpose(learned_delta, [0, 3, 1, 2])
        learned_delta = tf.reshape(
            learned_delta, [batch_size, self.token_count, self.token_dim])
        learned_delta = learned_delta * mask_3d

        self.last_residual_ratio = (
            tf.linalg.global_norm([learned_delta]) /
            (tf.linalg.global_norm([masked_x]) + 1e-12))
        total_abs = tf.add_n([
            tf.reduce_sum(tf.abs(mixer.kernel)) for mixer in self.head_mixers
        ])
        diagonal_abs = tf.add_n([
            tf.reduce_sum(tf.abs(tf.linalg.diag_part(mixer.kernel)))
            for mixer in self.head_mixers
        ])
        self.last_offdiag_ratio = (total_abs - diagonal_abs) / (total_abs + 1e-12)
        return learned_delta


class SharedSwiGLU(tf.keras.layers.Layer):
    """One FFN shared by all semantic tokens; domain identity is explicit."""

    def __init__(self, token_dim, hidden_ratio=2, **kwargs):
        super().__init__(**kwargs)
        hidden_dim = token_dim * hidden_ratio
        self.gate_dense = tf.keras.layers.Dense(hidden_dim, name='gate')
        self.up_dense = tf.keras.layers.Dense(hidden_dim, name='up')
        self.down_dense = tf.keras.layers.Dense(token_dim, name='down')

    def call(self, x):
        gate = tf.nn.swish(self.gate_dense(x))
        up = self.up_dense(x)
        return self.down_dense(gate * up)


class SemanticDomainRankMixerBlock(tf.keras.layers.Layer):
    def __init__(self, token_count, token_dim, num_heads=8, hidden_ratio=2, **kwargs):
        super().__init__(**kwargs)
        self.token_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.token_mixing = HeadedTokenAxisMixing(
            token_count, token_dim, num_heads=num_heads, name='headed_token_mixing')
        self.ffn_ln = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)
        self.shared_swiglu = SharedSwiGLU(
            token_dim, hidden_ratio=hidden_ratio, name='shared_swiglu')

    def call(self, x, token_mask):
        mask_3d = tf.expand_dims(tf.cast(token_mask, x.dtype), -1)
        token_delta = self.token_mixing(self.token_ln(x), token_mask)
        x = (x + token_delta) * mask_3d
        ffn_delta = self.shared_swiglu(self.ffn_ln(x))
        return (x + ffn_delta) * mask_3d


class SemanticDomainRankMixer(tf.keras.layers.Layer):
    def __init__(self, token_count=193, token_dim=256, num_heads=8,
                 num_blocks=2, hidden_ratio=2, **kwargs):
        super().__init__(**kwargs)
        if token_count != 193:
            raise ValueError("this experiment requires exactly 193 semantic tokens")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.t = token_count
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.blocks = [
            SemanticDomainRankMixerBlock(
                token_count, token_dim, num_heads=num_heads,
                hidden_ratio=hidden_ratio, name='semantic_block_%d' % block_index)
            for block_index in range(num_blocks)
        ]
        logger.info(
            "SemanticDomainRankMixer initialized: tokens=%d dim=%d heads=%d blocks=%d" %
            (token_count, token_dim, num_heads, num_blocks))

    def call(self, tokens, token_mask):
        if tokens.shape.rank != 3:
            raise ValueError("semantic tokens must have rank 3 [B,T,D]")
        if tokens.shape[1] is not None and tokens.shape[1] != self.t:
            raise ValueError("semantic token axis must be exactly %d" % self.t)
        x = tokens * tf.expand_dims(tf.cast(token_mask, tokens.dtype), -1)
        for block in self.blocks:
            x = block(x, token_mask)
        mask_3d = tf.expand_dims(tf.cast(token_mask, x.dtype), -1)
        denominator = tf.reduce_sum(mask_3d, axis=1)
        return tf.reduce_sum(x * mask_3d, axis=1) / tf.maximum(denominator, 1.0)

    def diagnostics(self, token_mask):
        residual_ratio = tf.reduce_mean(tf.stack([
            block.token_mixing.last_residual_ratio for block in self.blocks
        ]))
        offdiag_ratio = tf.reduce_mean(tf.stack([
            block.token_mixing.last_offdiag_ratio for block in self.blocks
        ]))
        active_tokens = tf.reduce_mean(tf.reduce_sum(
            tf.cast(token_mask, tf.float32), axis=1))
        active_fraction = active_tokens / float(self.t)
        return residual_ratio, offdiag_ratio, active_tokens, active_fraction
