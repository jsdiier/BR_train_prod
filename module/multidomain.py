import tensorflow as tf


class MXInputAdapter(tf.keras.layers.Layer):
    """Independent MX feature namespace projected into the shared latent space."""

    def __init__(self, slot_ids, feature_size, num_buckets, embedding_dim,
                 latent_dim, **kwargs):
        super(MXInputAdapter, self).__init__(name='mx_input_adapter', **kwargs)
        self.slot_ids = list(slot_ids)
        self.num_slots = len(self.slot_ids)
        self.feature_size = feature_size
        self.embedding_dim = embedding_dim

        self.slot_id_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(self.slot_ids, dtype=tf.int32),
                values=tf.range(self.num_slots, dtype=tf.int32)),
            default_value=-1,
            name='mx_slot_id_table')
        self.fid_table = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.int64,
            value_dtype=tf.int64,
            default_value=-1,
            empty_key=0,
            deleted_key=-1,
            initial_num_buckets=num_buckets,
            name='mx_fid_table')
        self.fid_counter = tf.Variable(
            1, dtype=tf.int64, trainable=False, name='mx_fid_counter')
        self.embedding = tf.keras.layers.Embedding(
            feature_size, embedding_dim, name='mx_feature_embedding')
        self.slot_context = tf.keras.layers.Embedding(
            self.num_slots, embedding_dim - 1, name='mx_slot_context')
        self.projection = tf.keras.layers.Dense(
            latent_dim, activation=tf.nn.swish, name='mx_latent_projection')
        self.projection_ln = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-5, name='mx_latent_ln')

    @staticmethod
    def _to_dense(values, dtype):
        if isinstance(values, tf.SparseTensor):
            values = tf.sparse.to_dense(values)
        return tf.cast(values, dtype)

    def _lookup_or_insert(self, fids):
        flat = tf.reshape(fids, [-1])
        unique_ids, inverse = tf.unique(flat)
        mapped = self.fid_table.lookup(unique_ids)
        missing = tf.equal(mapped, -1)

        def insert_missing():
            missing_indices = tf.where(missing)
            missing_ids = tf.gather_nd(unique_ids, missing_indices)
            count = tf.shape(missing_ids)[0]
            start = self.fid_counter.read_value()
            new_values = tf.range(start, start + tf.cast(count, tf.int64))
            capacity_check = tf.debugging.assert_less_equal(
                start + tf.cast(count, tf.int64),
                tf.cast(self.feature_size, tf.int64),
                message='MX fid embedding capacity exhausted')
            with tf.control_dependencies([capacity_check]):
                self.fid_table.insert(missing_ids, new_values)
                self.fid_counter.assign_add(tf.cast(count, tf.int64))
            return tf.tensor_scatter_nd_update(mapped, missing_indices, new_values)

        mapped = tf.cond(
            tf.reduce_any(missing), insert_missing, lambda: mapped)
        return tf.reshape(tf.gather(mapped, inverse), tf.shape(fids))

    def call(self, inputs, training=False):
        sids, fids = inputs
        sid_list = self._to_dense(sids, tf.int32)
        fid_list = self._to_dense(fids, tf.int64)
        batch_size = tf.shape(sid_list)[0]
        width = tf.shape(sid_list)[1]

        flat_sids = tf.reshape(sid_list, [-1])
        flat_fids = tf.reshape(fid_list, [-1])
        slot_indices = self.slot_id_table.lookup(flat_sids)
        valid = tf.logical_and(
            tf.not_equal(slot_indices, -1), tf.not_equal(flat_fids, 0))
        valid_slots = tf.boolean_mask(slot_indices, valid)
        valid_fids = tf.boolean_mask(flat_fids, valid)
        valid_slot_check = tf.debugging.assert_positive(
            tf.size(valid_fids), message='MX batch contains no configured slots')
        with tf.control_dependencies([valid_slot_check]):
            valid_fids = tf.identity(valid_fids)
            mapped_fids = (self._lookup_or_insert(valid_fids) if training
                           else self.fid_table.lookup(valid_fids))
        in_capacity = tf.logical_and(
            mapped_fids > 0,
            mapped_fids < tf.cast(self.feature_size, mapped_fids.dtype))
        mapped_fids = tf.boolean_mask(mapped_fids, in_capacity)
        valid_slots = tf.boolean_mask(valid_slots, in_capacity)

        batch_ids = tf.repeat(tf.range(batch_size), width)
        valid_batch_ids = tf.boolean_mask(batch_ids, valid)
        valid_batch_ids = tf.boolean_mask(valid_batch_ids, in_capacity)
        segment_ids = valid_batch_ids * self.num_slots + valid_slots

        embeddings = self.embedding(mapped_fids)
        pooled_flat = tf.math.unsorted_segment_sum(
            embeddings, segment_ids, batch_size * self.num_slots)
        pooled = tf.reshape(
            pooled_flat, [batch_size, self.num_slots, self.embedding_dim])

        counts_flat = tf.math.unsorted_segment_sum(
            tf.ones_like(segment_ids, dtype=tf.float32),
            segment_ids, batch_size * self.num_slots)
        slot_mask = tf.reshape(
            counts_flat > 0, [batch_size, self.num_slots])
        slot_mask_f = tf.cast(slot_mask, tf.float32)

        lr = tf.reduce_sum(pooled[:, :, 0], axis=1, keepdims=True)
        fm_embeddings = pooled[:, :, 1:]
        square_sum = tf.square(tf.reduce_sum(fm_embeddings, axis=1))
        sum_square = tf.reduce_sum(tf.square(fm_embeddings), axis=1)
        fm = 0.5 * (square_sum - sum_square)

        slot_ids = tf.range(self.num_slots, dtype=tf.int32)
        slot_context = self.slot_context(slot_ids)
        tokens = fm_embeddings + tf.expand_dims(slot_context, axis=0)
        mask_3d = tf.expand_dims(slot_mask_f, axis=-1)
        present = tf.reduce_sum(slot_mask_f, axis=1, keepdims=True)
        token_mean = tf.reduce_sum(tokens * mask_3d, axis=1) / (present + 1e-8)
        masked_tokens = tf.where(
            tf.cast(mask_3d, tf.bool), tokens,
            tf.fill(tf.shape(tokens), tf.cast(-1e9, tokens.dtype)))
        token_max = tf.reduce_max(masked_tokens, axis=1)
        token_max = tf.where(present > 0, token_max, tf.zeros_like(token_max))
        latent = self.projection(tf.concat([token_mean, token_max], axis=-1))
        latent = self.projection_ln(latent)
        return tf.concat([lr, fm, latent], axis=-1)

    def observed_fid_count(self):
        return self.fid_counter.read_value() - 1
