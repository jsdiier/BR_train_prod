import tensorflow as tf


class EMAShadow(tf.Module):
    """Trackable EMA state that survives checkpoint and process boundaries."""

    def __init__(self, model_weights, decay=0.999, name="ema_shadow"):
        super(EMAShadow, self).__init__(name=name)
        self.decay = tf.Variable(
            decay, trainable=False, dtype=tf.float32, name="decay"
        )
        self.values = []
        for index, weight in enumerate(model_weights):
            self.values.append(
                tf.Variable(
                    weight,
                    trainable=False,
                    name="shadow_%05d" % index,
                )
            )

    @tf.function(experimental_relax_shapes=True)
    def update(self, model_weights):
        one_minus_decay = 1.0 - self.decay
        for shadow, weight in zip(self.values, model_weights):
            shadow.assign(self.decay * shadow + one_minus_decay * weight)

    def assign_to(self, model_weights):
        if len(self.values) != len(model_weights):
            raise ValueError(
                "EMA/model variable count mismatch: %d != %d"
                % (len(self.values), len(model_weights))
            )
        for shadow, weight in zip(self.values, model_weights):
            weight.assign(shadow)
