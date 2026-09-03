import tensorflow as tf


class SelectiveAdamW(tf.keras.optimizers.Adam):
    """Adam plus decoupled weight decay applied only to Dense-style kernels."""

    def __init__(self, decoupled_weight_decay=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.decoupled_weight_decay = float(decoupled_weight_decay)

    @staticmethod
    def should_decay(variable):
        leaf_name = variable.name.lower().split('/')[-1]
        return leaf_name in ('kernel', 'kernel:0')

    def decay_audit(self, variables):
        decayed = [v.name for v in variables if self.should_decay(v)]
        excluded = [v.name for v in variables if not self.should_decay(v)]
        forbidden = [name for name in decayed if any(token in name.lower() for token in (
            'embedding', 'emb_fm', 'emb_din_ads', 'bias', 'beta', 'gamma'))]
        if forbidden:
            raise ValueError('forbidden variables selected for AdamW decay: %s' % forbidden)
        if not decayed:
            raise ValueError('AdamW decay whitelist is empty')
        return decayed, excluded

    def apply_gradients(self, grads_and_vars, name=None, **kwargs):
        grads_and_vars = list(grads_and_vars)
        decay_ops = []
        for gradient, variable in grads_and_vars:
            if gradient is None or not self.should_decay(variable):
                continue
            learning_rate = self._decayed_lr(variable.dtype.base_dtype)
            decay = (
                tf.cast(learning_rate, variable.dtype) *
                tf.cast(self.decoupled_weight_decay, variable.dtype) *
                variable)
            decay_ops.append(variable.assign_sub(decay))
        with tf.control_dependencies(decay_ops):
            return super().apply_gradients(
                grads_and_vars, name=name, **kwargs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'decoupled_weight_decay': self.decoupled_weight_decay,
        })
        return config
