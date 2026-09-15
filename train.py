import utils as ut
import argparse
import copy
import sys, os
import numpy as np
import tensorflow as tf
import datetime
from datetime import timedelta
import model_conf
from model import Model
from tensorflow.keras import regularizers
import time
import hashlib
from sklearn import metrics
from wandb_monitor import WandbMonitor

class Learner:
    def __init__(self):
        self.model = None
        self.wandb_monitor = WandbMonitor()
        self.wandb_log_interval = max(1, int(os.environ.get('WANDB_LOG_INTERVAL', '100')))
        self.wandb_param_interval = max(0, int(os.environ.get('WANDB_PARAM_INTERVAL', '1000')))
        self.wandb_task_grad_interval = max(0, int(os.environ.get('WANDB_TASK_GRAD_INTERVAL', '0')))
        self.loss_window_sum = tf.Variable(tf.zeros([5], tf.float32), trainable=False)
        self.loss_window_square_sum = tf.Variable(tf.zeros([5], tf.float32), trainable=False)
        self.loss_window_count = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def set_training_mode(self, enable_training, is_save_model):
        self.model.training = enable_training
        self.model.is_save_model = is_save_model

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, feat, buy_weight=1.0, cat_weight=1.0, click_weight=1.0, ext_weight=1.0,
                   collect_grad_stats=False, collect_param_stats=False,
                   collect_task_grad_stats=False):
        model = self.model
        with tf.GradientTape(persistent=collect_task_grad_stats) as tape:
            pred_buy, pred_cat, pred_click, pred_ext = model([feat['fea_ids'], feat['fea_vals']])

            loss_buy = model.loss_bc(tf.expand_dims(feat['cvr_label'], 1), pred_buy)
            loss_cat = model.loss_bc(tf.expand_dims(feat['cat_label'], 1), pred_cat)
            loss_click = model.loss_bc(tf.expand_dims(feat['clk_label'], 1), pred_click)
            loss_ext = model.loss_bc(tf.expand_dims(feat['ext_label'], 1), pred_ext)

            final_loss = loss_buy * buy_weight + loss_cat * cat_weight + loss_click * click_weight + loss_ext * ext_weight

            loss_means = tf.stack([
                tf.reduce_mean(loss_buy),
                tf.reduce_mean(loss_cat),
                tf.reduce_mean(loss_click),
                tf.reduce_mean(loss_ext),
                tf.reduce_mean(final_loss),
            ])
            self.loss_window_sum.assign_add(loss_means)
            self.loss_window_square_sum.assign_add(tf.square(loss_means))
            self.loss_window_count.assign_add(1.0)

            gradients = tape.gradient(final_loss, model.trainable_weights)

        monitor_stats = self._empty_monitor_stats()
        if collect_grad_stats or collect_param_stats:
            monitor_stats.update(self._collect_module_stats(
                gradients,
                model.trainable_weights,
                tf.shape(feat['cvr_label'])[0],
                collect_grad_stats,
                collect_param_stats,
            ))
        if collect_task_grad_stats:
            monitor_stats.update(self._collect_task_gradient_stats(
                tape,
                [loss_buy, loss_cat, loss_click, loss_ext],
            ))
            del tape
        if collect_grad_stats or collect_param_stats or collect_task_grad_stats:
            monitor_stats.update(self._loss_window_stats())

        model.optimizer.apply_gradients(zip(gradients, model.trainable_weights))
        return (loss_buy, loss_cat, loss_click, loss_ext, final_loss,
                pred_buy, pred_cat, pred_click, pred_ext, monitor_stats)

    def _loss_window_stats(self):
        count = tf.maximum(self.loss_window_count, 1.0)
        mean = self.loss_window_sum / count
        variance = tf.maximum(self.loss_window_square_sum / count - tf.square(mean), 0.0)
        std = tf.sqrt(variance)
        names = ['buy', 'cat', 'click', 'ext', 'total']
        stats = {'loss/window_batches': self.loss_window_count}
        for index, name in enumerate(names):
            stats['loss_window/%s_mean' % name] = mean[index]
            stats['loss_window/%s_std' % name] = std[index]
        return stats

    def _reset_loss_window(self):
        self.loss_window_sum.assign(tf.zeros_like(self.loss_window_sum))
        self.loss_window_square_sum.assign(tf.zeros_like(self.loss_window_square_sum))
        self.loss_window_count.assign(0.0)

    @staticmethod
    def _gradient_values(gradient):
        if isinstance(gradient, tf.IndexedSlices):
            return gradient.values
        return gradient

    def _empty_monitor_stats(self):
        nan = tf.constant(float('nan'), dtype=tf.float32)
        keys = [
            'grad/global_norm_raw',
            'grad/global_norm_per_sample',
            'grad/max_abs',
            'grad/nonfinite_count',
            'grad/zero_fraction',
            'loss/window_batches',
        ]
        for name in ['buy', 'cat', 'click', 'ext', 'total']:
            keys.extend([
                'loss_window/%s_mean' % name,
                'loss_window/%s_std' % name,
            ])
        module_names = ['embedding', 'din', 'shared', 'buy_head', 'cat_head', 'click_head', 'ext_head']
        for name in module_names:
            keys.extend([
                'grad/module_%s_norm' % name,
                'parameter/module_%s_norm' % name,
                'gradient_to_parameter/module_%s' % name,
            ])
        task_names = ['buy', 'cat', 'click', 'ext']
        for name in task_names:
            keys.append('grad/task_%s_shared_norm' % name)
        for left_index, left in enumerate(task_names):
            for right in task_names[left_index + 1:]:
                keys.append('grad/cosine_shared_%s_%s' % (left, right))
        return {key: nan for key in keys}

    @staticmethod
    def _layer_variable_ids(layers):
        variable_ids = set()
        for layer in layers:
            if layer is None:
                continue
            for variable in layer.trainable_variables:
                variable_ids.add(id(variable))
        return variable_ids

    def _module_variable_groups(self):
        model = self.model
        embedding_ids = self._layer_variable_ids([model.emb_fm, model.emb_din_ads])
        din_ids = self._layer_variable_ids([
            model.seq_click_attention_layer,
            model.seq_pay_attention_layer,
            model.seq_12h_click_cate_id_attention_layer,
            model.attention_layer_search_long_pay,
            model.attention_layer_search_long_clk,
            model.attention_layer_search_long_query,
            model.pay_seq_ln,
            model.pay_seq_proj,
            model.pay_seq_combine,
            model.clk_seq_ln,
            model.clk_seq_proj,
            model.clk_seq_combine,
            model.query_seq_ln,
            model.query_seq_proj,
            model.query_seq_combine,
        ])
        buy_ids = self._layer_variable_ids([model.buy_tower, model.dense_concat])
        cat_ids = self._layer_variable_ids([model.cat_tower, model.dense_concat1])
        click_ids = self._layer_variable_ids([model.click_tower, model.dense_concat2])
        ext_ids = self._layer_variable_ids([model.ext_tower, model.dense_concat3])
        assigned_ids = embedding_ids | din_ids | buy_ids | cat_ids | click_ids | ext_ids
        shared_ids = set(id(variable) for variable in model.trainable_weights) - assigned_ids
        return {
            'embedding': embedding_ids,
            'din': din_ids,
            'shared': shared_ids,
            'buy_head': buy_ids,
            'cat_head': cat_ids,
            'click_head': click_ids,
            'ext_head': ext_ids,
        }

    def _collect_module_stats(self, gradients, variables, batch_size,
                              collect_grad_stats, collect_param_stats):
        stats = {}
        valid_pairs = [
            (gradient, variable)
            for gradient, variable in zip(gradients, variables)
            if gradient is not None
        ]
        valid_gradients = [pair[0] for pair in valid_pairs]
        batch_size_float = tf.cast(tf.maximum(batch_size, 1), tf.float32)

        if collect_grad_stats:
            gradient_values = [self._gradient_values(gradient) for gradient in valid_gradients]
            global_norm = tf.linalg.global_norm(valid_gradients)
            nonfinite_count = tf.add_n([
                tf.reduce_sum(tf.cast(tf.logical_not(tf.math.is_finite(value)), tf.float32))
                for value in gradient_values
            ]) if gradient_values else tf.constant(0.0, tf.float32)
            zero_count = tf.add_n([
                tf.reduce_sum(tf.cast(tf.equal(value, 0), tf.float32))
                for value in gradient_values
            ]) if gradient_values else tf.constant(0.0, tf.float32)
            value_count = tf.add_n([
                tf.cast(tf.size(value), tf.float32)
                for value in gradient_values
            ]) if gradient_values else tf.constant(0.0, tf.float32)
            max_abs = tf.reduce_max(tf.stack([
                tf.reduce_max(tf.abs(value)) for value in gradient_values
            ])) if gradient_values else tf.constant(0.0, tf.float32)
            stats.update({
                'grad/global_norm_raw': global_norm,
                'grad/global_norm_per_sample': global_norm / batch_size_float,
                'grad/max_abs': max_abs,
                'grad/nonfinite_count': nonfinite_count,
                'grad/zero_fraction': zero_count / tf.maximum(value_count, 1.0),
            })

        groups = self._module_variable_groups()
        for name, variable_ids in groups.items():
            group_pairs = [pair for pair in valid_pairs if id(pair[1]) in variable_ids]
            group_gradients = [pair[0] for pair in group_pairs]
            group_variables = [pair[1] for pair in group_pairs]
            grad_norm = None
            param_norm = None
            if collect_grad_stats:
                grad_norm = (tf.linalg.global_norm(group_gradients) if group_gradients
                             else tf.constant(0.0, tf.float32))
                stats['grad/module_%s_norm' % name] = grad_norm
            if collect_param_stats:
                param_norm = (tf.linalg.global_norm(group_variables) if group_variables
                              else tf.constant(0.0, tf.float32))
                stats['parameter/module_%s_norm' % name] = param_norm
            if collect_grad_stats and collect_param_stats:
                stats['gradient_to_parameter/module_%s' % name] = (
                    grad_norm / tf.maximum(param_norm, tf.constant(1e-12, tf.float32))
                )
        return stats

    @staticmethod
    def _gradient_dot(left_gradients, right_gradients):
        products = []
        for left, right in zip(left_gradients, right_gradients):
            if left is None or right is None:
                continue
            left_value = left.values if isinstance(left, tf.IndexedSlices) else left
            right_value = right.values if isinstance(right, tf.IndexedSlices) else right
            products.append(tf.reduce_sum(left_value * right_value))
        if not products:
            return tf.constant(0.0, tf.float32)
        return tf.add_n(products)

    def _collect_task_gradient_stats(self, tape, task_losses):
        groups = self._module_variable_groups()
        shared_variables = [
            variable for variable in self.model.trainable_weights
            if id(variable) in groups['shared']
        ]
        task_names = ['buy', 'cat', 'click', 'ext']
        task_gradients = {}
        task_norms = {}
        stats = {}
        for name, loss in zip(task_names, task_losses):
            # GradientTape sums a non-scalar target, matching the baseline's
            # final-loss gradient semantics without creating an unrecorded op.
            task_gradients[name] = tape.gradient(loss, shared_variables)
            valid = [gradient for gradient in task_gradients[name] if gradient is not None]
            task_norms[name] = (tf.linalg.global_norm(valid) if valid
                                else tf.constant(0.0, tf.float32))
            stats['grad/task_%s_shared_norm' % name] = task_norms[name]
        for left_index, left in enumerate(task_names):
            for right in task_names[left_index + 1:]:
                denominator = tf.maximum(
                    task_norms[left] * task_norms[right],
                    tf.constant(1e-12, tf.float32),
                )
                stats['grad/cosine_shared_%s_%s' % (left, right)] = (
                    self._gradient_dot(task_gradients[left], task_gradients[right]) / denominator
                )
        return stats

    def _date_range(self, start, end):
        """返回 [start, end] 闭区间内的所有天(YYYYMMDD 字符串,升序)"""
        d = datetime.datetime.strptime(start, "%Y%m%d")
        end_d = datetime.datetime.strptime(end, "%Y%m%d")
        days = []
        while d <= end_d:
            days.append(d.strftime("%Y%m%d"))
            d += timedelta(1)
        return days

    def get_day_files(self, data_arg, day):
        """取某一天的数据文件。data_arg 可以是逗号分隔的天目录,也可以是基础目录。"""
        base_paths = [p.rstrip('/') for p in str(data_arg).split(',') if p]
        files = []
        for bp in base_paths:
            if bp.endswith(day):
                files += tf.io.gfile.glob(bp + "/part*")
            else:
                files += tf.io.gfile.glob("%s/%s/part*" % (bp, day))
        return sorted(set(files))

    def train(self, data_arg, start_day, end_day, model_path=None, data_path=None, dump_serving_model=True):
        if self.model is None:
            self.model = Model(training=True)
        model = self.model

        train_writer = None
        tensorboard_dir = "./log/tensorboard_data_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if tensorboard_dir:
            train_writer = tf.summary.create_file_writer(
                os.path.join(tensorboard_dir, "train"), max_queue=1000
            )
            model.set_summary_writer(train_writer, histogram_freq=100)
            print('TensorBoard enabled:', tensorboard_dir)

        days = self._date_range(start_day, end_day)
        print("train days [%s, %s], total %d days" % (start_day, end_day, len(days)))

        batch_size = model_conf.batch_size
        shuffle_size = batch_size * 10

        #load ckpt (需要先跑一个 batch 建好变量再 restore)
        ckpt_path = model_path or self.get_model_checkpoint_from_file(model_conf.done_file_path)
        if ckpt_path is not None:
            print("load model from checkpoint:", ckpt_path)
            ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)

            probe_files = self.get_day_files(data_arg, days[0])
            probe_ds = ut.ReadTFRecordV2(probe_files, shuffle_size=1, batch_size=batch_size, fetch_size=1, num_parallel=10)
            first_batch = next(iter(probe_ds))
            _ = model([first_batch['fea_ids'], first_batch['fea_vals']])

            dummy_grad = [tf.zeros_like(v) for v in model.trainable_variables]
            model.optimizer.apply_gradients(zip(dummy_grad, model.trainable_variables))

            ckpt.restore(tf.train.latest_checkpoint(ckpt_path)).assert_consumed()
            print("Restored optimizer step: ", model.optimizer.iterations.numpy())
            print("load checkpoint path: ", ckpt_path)

        self.wandb_monitor.start(
            config={
                'batch_size': model_conf.batch_size,
                'learning_rate': model_conf.learning_rate,
                'l2_reg': model_conf.l2_reg,
                'feature_size': model_conf.feature_size,
                'num_buckets': model_conf.num_buckets,
                'fm_emb_size': model_conf.fm_emb_size,
                'din_emb_size': model_conf.din_emb_size,
                'train_hdfs': data_arg,
                'checkpoint_restore_path': ckpt_path or 'fresh_initialization',
                'wandb_log_interval': self.wandb_log_interval,
                'wandb_param_interval': self.wandb_param_interval,
                'wandb_task_grad_interval': self.wandb_task_grad_interval,
                'task_gradient_diagnostics_enabled': self.wandb_task_grad_interval > 0,
                'slot_count': len(model_conf.all_slot_ids),
                'slot_config_hash_sha256': hashlib.sha256(
                    ','.join(str(slot_id) for slot_id in model_conf.all_slot_ids).encode('utf-8')
                ).hexdigest(),
            },
            start_day=start_day,
            end_day=end_day,
        )

        #每天训练完直接算指标,结果按天写到 metrics 文件(不落 pred/label 明细,省内存/磁盘)
        out_dir = model_conf.local_model_dir
        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir)
            except Exception:
                pass
        metric_path = os.path.join(out_dir, 'metrics_by_day.txt')
        task_names = ['buy', 'cat', 'click', 'ext']

        mfout = open(metric_path, 'a')
        mfout.write('\t'.join(['day', 'task', 'n', 'auc', 'gauc', 'mae', 'pos_rate']) + '\n')

        #多天累计统计:各任务正样本数 [buy, cat, click, ext] 和总样本数
        self.pos = np.zeros(len(task_names))
        self.cnt = 0
        self.gstep = 0

        print('training...')
        for idx, day in enumerate(days):
            files = self.get_day_files(data_arg, day)
            if not files:
                print(datetime.datetime.now(), "day %s: no files found, skip" % day)
                continue
            print(datetime.datetime.now(), "==== start day %s (%d/%d), %d files ====" % (
                day, idx + 1, len(days), len(files)))

            #train one pass over this day(训练完当天直接算指标写入 metrics 文件)
            self.set_training_mode(True, False)
            ds = ut.ReadTFRecordV2(files, shuffle_size=shuffle_size, batch_size=batch_size, fetch_size=10, num_parallel=10)
            ds = ds.apply(tf.data.experimental.ignore_errors())
            self.train_one_day(ds, day, train_writer, mfout)

            #当天训练的 summary 落盘
            if train_writer is not None:
                train_writer.flush()

            #每 N 天保存一个 ckpt(最后一天也保存)
            is_last = (idx == len(days) - 1)
            if (idx + 1) % model_conf.ckpt_save_days == 0 or is_last:
                self.save_checkpoint(day)

        mfout.close()
        print(datetime.datetime.now(), "metrics written to %s" % metric_path)

        #导出 serving 模型。滚动评估只需要 checkpoint，可显式关闭，避免每天重复导出。
        if dump_serving_model:
            self.set_training_mode(False, True)
            self.dump_serving_model(end_day, 0)
            self.set_training_mode(True, False)
        self.wandb_monitor.finish()

    @staticmethod
    def _tensor_float(value):
        if hasattr(value, 'numpy'):
            value = value.numpy()
        array = np.asarray(value)
        return float(array.reshape([-1])[0])

    def _wandb_step_payload(self, day, losses, predictions, monitor_stats,
                            elapsed_seconds, samples_in_window):
        model = self.model
        task_names = ['buy', 'cat', 'click', 'ext']
        payload = {
            'optimizer_step': int(model.optimizer.iterations.numpy()),
            'progress/train_day': int(day),
            'progress/process_step': int(self.gstep),
            'progress/cumulative_samples': int(self.cnt),
            'performance/window_seconds': float(elapsed_seconds),
            'performance/examples_per_second': (
                float(samples_in_window) / max(float(elapsed_seconds), 1e-12)
            ),
            'optimizer/learning_rate': self._tensor_float(
                model.lr_schedule(model.optimizer.iterations)
            ),
        }

        loss_names = ['buy', 'cat', 'click', 'ext', 'total']
        for name, value in zip(loss_names, losses):
            array = np.asarray(value.numpy(), dtype=np.float64).reshape([-1])
            payload['loss/%s' % name] = float(np.mean(array))
            payload['numeric/loss_%s_nonfinite_count' % name] = int(
                np.count_nonzero(~np.isfinite(array))
            )

        for task_index, (name, prediction) in enumerate(zip(task_names, predictions)):
            array = np.asarray(prediction.numpy(), dtype=np.float64).reshape([-1])
            finite = array[np.isfinite(array)]
            payload['numeric/prediction_%s_nonfinite_count' % name] = int(
                array.size - finite.size
            )
            if finite.size:
                payload['prediction/%s_mean' % name] = float(np.mean(finite))
                payload['prediction/%s_std' % name] = float(np.std(finite))
                payload['prediction/%s_p01' % name] = float(np.percentile(finite, 1))
                payload['prediction/%s_p50' % name] = float(np.percentile(finite, 50))
                payload['prediction/%s_p99' % name] = float(np.percentile(finite, 99))
                payload['prediction/%s_below_0_01' % name] = float(np.mean(finite < 0.01))
                payload['prediction/%s_above_0_99' % name] = float(np.mean(finite > 0.99))
            payload['data/pos_rate_%s' % name] = float(
                self.pos[task_index] / max(self.cnt, 1)
            )

        for name, value in monitor_stats.items():
            numeric_value = self._tensor_float(value)
            if np.isfinite(numeric_value):
                payload[name] = numeric_value
        return payload

    def train_one_day(self, train_data, day, train_writer, mfout=None):
        model = self.model

        uid_index = model_conf.uid_add_info_index
        field_num = model_conf.add_info_field_num
        mod_threshold = model_conf.eval_uid_ratio * 100  # uid % 100 < 此值 则命中(0.01 -> 1)
        label_keys = ['cvr_label', 'cat_label', 'clk_label', 'ext_label']

        #当天采样子集缓存在内存(只保留当天,算完即释放,避免落大文件/OOM)
        eval_uids = []
        eval_labels = [[] for _ in label_keys]
        eval_preds = [[] for _ in label_keys]

        n_sampled = 0
        step = -1
        monitor_window_start = time.time()
        monitor_window_samples = 0
        for step, feat in enumerate(train_data):
            label_arrs = [np.reshape(feat[k].numpy(), [-1]) for k in label_keys]
            self.cnt += label_arrs[0].shape[0]
            self.pos += [a.sum() for a in label_arrs]
            monitor_window_samples += label_arrs[0].shape[0]

            next_process_step = self.gstep + 1
            collect_grad_stats = next_process_step % self.wandb_log_interval == 0
            collect_param_stats = (
                self.wandb_param_interval > 0
                and next_process_step % self.wandb_param_interval == 0
            )
            collect_task_grad_stats = (
                self.wandb_task_grad_interval > 0
                and next_process_step % self.wandb_task_grad_interval == 0
            )
            (loss_buy, loss_cat, loss_click, loss_ext, final_loss,
             pred_buy, pred_cat, pred_click, pred_ext, monitor_stats) = self.train_step(
                feat,
                collect_grad_stats=collect_grad_stats,
                collect_param_stats=collect_param_stats,
                collect_task_grad_stats=collect_task_grad_stats,
            )

            #收集 uid 采样子集的 pred/label 到内存,当天训完直接算指标
            if mfout is not None:
                uids = feat['add_info_list'].values.numpy()[uid_index::field_num]
                #向量化:整批一次算出命中下标(避免整批逐样本 Python 循环)
                try:
                    uid_int = uids.astype('S').astype(np.int64)
                    sel = np.where((uid_int % 100) < mod_threshold)[0]
                except Exception:
                    sel = np.array([], dtype=np.int64)
                if sel.size > 0:  # 命中才把 pred 拉到 host
                    pred_arrs = [
                        np.reshape(pred_buy.numpy(), [-1]),
                        np.reshape(pred_cat.numpy(), [-1]),
                        np.reshape(pred_click.numpy(), [-1]),
                        np.reshape(pred_ext.numpy(), [-1]),
                    ]
                    n_sampled += sel.size
                    eval_uids.extend(uids[j].decode('utf-8', 'ignore') for j in sel)
                    for t_idx in range(len(label_keys)):
                        eval_labels[t_idx].extend(label_arrs[t_idx][sel].tolist())
                        eval_preds[t_idx].extend(pred_arrs[t_idx][sel].tolist())

            self.gstep += 1
            if collect_grad_stats or collect_param_stats or collect_task_grad_stats:
                monitor_now = time.time()
                payload = self._wandb_step_payload(
                    day,
                    [loss_buy, loss_cat, loss_click, loss_ext, final_loss],
                    [pred_buy, pred_cat, pred_click, pred_ext],
                    monitor_stats,
                    monitor_now - monitor_window_start,
                    monitor_window_samples,
                )
                self.wandb_monitor.log(payload)
                self._reset_loss_window()
                monitor_window_start = monitor_now
                monitor_window_samples = 0

            if train_writer is not None and self.gstep % 100 == 0:
                global_step = model.optimizer.iterations.numpy()  # 只在打点时同步一次,作 x 轴
                with train_writer.as_default():
                    tf.summary.scalar('loss_buy', tf.reduce_mean(loss_buy), step=global_step)
                    tf.summary.scalar('loss_cat', tf.reduce_mean(loss_cat), step=global_step)
                    tf.summary.scalar('loss_click', tf.reduce_mean(loss_click), step=global_step)
                    tf.summary.scalar('loss_ext', tf.reduce_mean(loss_ext), step=global_step)
                    tf.summary.scalar('loss/total', tf.reduce_mean(final_loss), step=global_step)

                    tf.summary.scalar('data/pos_rate_buy', self.pos[0] / max(self.cnt, 1), step=global_step)
                    tf.summary.scalar('data/pos_rate_click', self.pos[2] / max(self.cnt, 1), step=global_step)

                print(datetime.datetime.now(),
                        "day %s steps: %d, buy loss: %04f, pos: %d, cnt: %d,  cat loss: %04f, cat pos: %d,  click loss: %04f, click pos: %d,  ext loss: %04f, ext pos: %d, sampled: %d" % (
                        day, self.gstep, tf.reduce_mean(loss_buy), self.pos[0], self.cnt, tf.reduce_mean(loss_cat), self.pos[1],
                        tf.reduce_mean(loss_click), self.pos[2], tf.reduce_mean(loss_ext), self.pos[3], n_sampled))

        if step < 0:
            print(datetime.datetime.now(), "day %s finish, no batches" % day)
            return

        #当天训练完成后直接算 auc/gauc/mae,把结果写到 metrics 文件
        if mfout is not None:
            self._write_day_metrics(day, eval_uids, eval_labels, eval_preds, mfout, train_writer)

    def _write_day_metrics(self, day, uids, labels_by_task, preds_by_task, mfout, train_writer=None):
        """用当天内存里的采样子集算 auc/gauc/mae,结果写入 metrics 文件(+TensorBoard)。"""
        task_names = ['buy', 'cat', 'click', 'ext']
        try:
            step = int(day)  # 用日期做 tensorboard step(按天单调递增)
        except Exception:
            step = 0
        wandb_day_metrics = {
            'optimizer_step': int(self.model.optimizer.iterations.numpy()),
            'progress/train_day': int(day),
        }
        for t_idx, t in enumerate(task_names):
            labels = np.asarray(labels_by_task[t_idx], dtype=np.float32)
            preds = np.asarray(preds_by_task[t_idx], dtype=np.float32)
            n = labels.shape[0]
            if n == 0:
                print("METRIC day=%s task=%s: no samples" % (day, t))
                continue
            mae = float(np.mean(np.abs(labels - preds)))
            try:
                auc = metrics.roc_auc_score(labels, preds)
            except Exception as e:
                auc = float('nan')
                print("METRIC day=%s task=%s auc failed: %s" % (day, t, e))
            try:
                gauc, _ = ut.cal_group_auc(labels, preds, uids)
            except Exception:
                gauc = float('nan')
            pos_rate = float(np.mean(labels))
            wandb_day_metrics['daily_eval/%s_n' % t] = int(n)
            wandb_day_metrics['daily_eval/%s_auc' % t] = float(auc)
            wandb_day_metrics['daily_eval/%s_gauc' % t] = float(gauc)
            wandb_day_metrics['daily_eval/%s_mae' % t] = float(mae)
            wandb_day_metrics['daily_eval/%s_pos_rate' % t] = pos_rate
            print("METRIC day=%s task=%s n=%d auc=%.4f gauc=%.4f mae=%.4f pos_rate=%.4f" % (
                day, t, n, auc, gauc, mae, pos_rate))
            mfout.write("%s\t%s\t%d\t%.4f\t%.4f\t%.4f\t%.4f\n" % (day, t, n, auc, gauc, mae, pos_rate))
            if train_writer is not None:
                with train_writer.as_default():
                    tf.summary.scalar('eval_auc/%s' % t, auc, step=step)
                    tf.summary.scalar('eval_gauc/%s' % t, gauc, step=step)
                    tf.summary.scalar('eval_mae/%s' % t, mae, step=step)
        mfout.flush()
        self.wandb_monitor.log(wandb_day_metrics)

    def save_checkpoint(self, day):
        model = self.model
        save_dir = "%s/checkpoints/%s/" % (model_conf.local_model_dir, day)
        export_dir = save_dir + "tfmodel"
        ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        ckpt.save(export_dir)

        done_dir = os.path.dirname(model_conf.done_file_path)
        if done_dir and not os.path.exists(done_dir):
            try:
                os.makedirs(done_dir)
            except Exception as e:
                print("Warning: Failed to create done_file directory %s:" % done_dir, e)
        try:
            with open(model_conf.done_file_path, 'a') as f:
                f.write(day + "\t" + save_dir + "\n")
        except Exception as e:
            print("Warning: Failed to write done file %s:" % model_conf.done_file_path, e)
        print(datetime.datetime.now(), "saved checkpoint for day %s -> %s" % (day, save_dir))

    def dump_serving_model(self, end_day, epo):
        if self.model is None:
            return

        train_model = self.model

        fid_keys, fid_values = train_model.fid_table.export()
        fid_keys_ads, fid_values_ads = train_model.fid_table_din_ads.export()

        fid_keys = tf.reshape(fid_keys, [-1])
        fid_values = tf.reshape(fid_values, [-1])

        fid_keys_ads = tf.reshape(fid_keys_ads, [-1])
        fid_values_ads = tf.reshape(fid_values_ads, [-1])

        serve_model = Model(
            training=False,
            pred=True,
            fid_kv=(fid_keys, fid_values),
            fid_ads_kv=(fid_keys_ads, fid_values_ads)
        )
        serve_model.compile(optimizer=self.model.optimizer, loss=self.model.loss_bc, metrics=['mae'])

        serve_model.training = False
        serve_model.is_save_model = True
        dummy_sids = tf.constant([[0] * model_conf.padding_size], tf.uint32)
        dummy_fids = tf.constant([[0] * model_conf.padding_size], tf.uint64)
        _ = serve_model([dummy_sids, dummy_fids])

        serve_model._set_inputs([
            tf.keras.Input(shape=(model_conf.padding_size,), dtype=tf.dtypes.uint32),
            tf.keras.Input(shape=(model_conf.padding_size,), dtype=tf.dtypes.uint64),
        ])

        #serve_model.load_weights(model_conf.model_path)
        serve_model.set_weights(train_model.get_weights())

        serve_model.save(
            'serving_model_%s/%s00' % (epo, end_day),
            save_format='tf'
        )

    def count_parameters(self, verbose=False):
        total_params = 0
        if self.model is None:
            print("Model not initialized")
            return
        for i,weight in enumerate(self.model.trainable_weights):
            shape = weight.shape.as_list()
            num_params = np.prod(shape)
            total_params += num_params
            if verbose:
                print("Layer "+str(i)+":"+str(weight.name)+" | Shape:"+str(shape)+" | params:"+str(num_params))
        print("Total trainable parameters:"+str(total_params))
        return total_params

    def get_model_checkpoint_from_file(self, done_file_path='model.done'):
        if not os.path.exists(done_file_path):
            print("model.done not exit in patch: ",done_file_path)
            return None

        with open(done_file_path, 'r') as f:
            lines = f.readlines()
            if not lines:
                print("model.done is null ")
                return None

            # read last line
            last_line = lines[-1].strip()
            if not last_line:
                print("model.done last line is null")
                return None

            # file format: checkpoint_day\tcheckpoint_path
            parts = last_line.split('\t')
            if len(parts) >= 2:
                ckpt_day = parts[0]
                ckpt_path = parts[1]
                print("load model checkpoint_path=%s, checkpoint_day=%s",ckpt_path, ckpt_day)
                return ckpt_path
            else:
                print("model.done last line format error")
                return None

if __name__ == "__main__":
    # init args and model
    parse = argparse.ArgumentParser(description='get input args')
    parse.add_argument('-data', type=str, help='input data files')
    parse.add_argument('-start_day', type=str, help='train start day')
    parse.add_argument('-end_day', type=str, help='train end day')
    parse.add_argument('-checkpoint_path', type=str, default=None,
                       help='explicit checkpoint directory; defaults to last entry in model.done')
    parse.add_argument('-dump_serving_model', type=int, default=1,
                       help='1: dump serving model after training; 0: checkpoint only')

    args = parse.parse_args()
    solver = Learner()

    # set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = model_conf.gpu_id
    print('CUDA_VISIBLE_DEVICES', os.environ['CUDA_VISIBLE_DEVICES'])
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # start training or testing
    if model_conf.train_mode == 'train':
        #按天 for 循环读取数据并训练(数据读取放在 train() 内部,逐天进行)
        print('start training, day by day from %s to %s' % (args.start_day, args.end_day))
        start_time = time.time()
        solver.train(args.data, model_path=args.checkpoint_path, data_path=args.data,
                     start_day=args.start_day, end_day=args.end_day,
                     dump_serving_model=bool(args.dump_serving_model))
        end_time2 = time.time()
        print('end training, using_time_training: ', end_time2 - start_time)
    else:
        pass
