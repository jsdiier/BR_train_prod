#!/usr/bin/env python3
import os
import sys
import tempfile
import unittest


sys.path.insert(0, os.path.dirname(__file__))

from validate_experiment import rolling_expectations, validate


class RollingExpectationsTest(unittest.TestCase):
    metric_lines = "\n".join(
        "test_%s auc:0.800000 gauc:0.700000 uauc:0.600000 "
        "size:100 loss:0.100000, pos: 10" % task
        for task in ("buy", "cat", "click", "ext")
    )

    def test_contiguous_window_uses_previous_day_checkpoint(self):
        checkpoints, logs = rolling_expectations(
            "20260724", "20260801", set(), set())
        self.assertEqual(
            checkpoints,
            ["20260724", "20260725", "20260726", "20260727",
             "20260728", "20260729", "20260730", "20260731"],
        )
        self.assertIn(("20260728", "20260729", "20260729"), logs)
        self.assertEqual(("20260731", "20260801", "20260801"), logs[-1])

    def test_missing_day_uses_latest_real_checkpoint(self):
        checkpoints, logs = rolling_expectations(
            "20260724", "20260801", {"20260728"}, {"20260728"})
        self.assertNotIn("20260728", checkpoints)
        self.assertNotIn(("20260727", "20260728", "20260728"), logs)
        self.assertIn(("20260727", "20260729", "20260729"), logs)
        self.assertEqual(7, len(logs))

    def test_validation_accepts_declared_missing_day(self):
        config = {
            "train_end_day": "20260720",
            "test_start_day": "20260721",
            "test_end_day": "20260724",
            "auto_test_start_ckpt_day": "20260724",
            "auto_test_end_day": "20260801",
            "allowed_missing_train_days": ["20260728"],
            "allowed_missing_test_days": ["20260728"],
            "require_inference_benchmark": True,
        }
        checkpoints, rolling_logs = rolling_expectations(
            "20260724", "20260801", {"20260728"}, {"20260728"})
        fixed_key = ("20260720", "20260721", "20260724")
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "log"))
            os.makedirs(os.path.join(root, "model"))
            for day in ["20260720"] + checkpoints:
                ckpt_dir = os.path.join(root, "model", "checkpoints", day)
                os.makedirs(ckpt_dir)
                for name in ("checkpoint", "tfmodel-1.index",
                             "tfmodel-1.data-00000-of-00001"):
                    open(os.path.join(ckpt_dir, name), "w").close()

            fixed_content = self.metric_lines + "\n" + "\n".join((
                "[INFERENCE_BENCHMARK] device:GPU batch_size:1024 "
                "warmup_batches:20 measure_batches:100 samples:102400",
                "[INFERENCE_BENCHMARK] model throughput_samples_s:1",
                "[INFERENCE_BENCHMARK] end_to_end throughput_samples_s:1",
            ))
            all_logs = [("fixed", fixed_key, fixed_content)] + [
                ("rolling", key, self.metric_lines) for key in rolling_logs]
            for prefix, key, content in all_logs:
                name = "%s_test_ckpt_%s_from_%s_to_%s_202608280000" % (
                    prefix, key[0], key[1], key[2])
                with open(os.path.join(root, "log", name), "w") as handle:
                    handle.write(content)

            summary = os.path.join(root, "model", "rolling_metrics.tsv")
            with open(summary, "w") as handle:
                handle.write("checkpoint_day\ttest_start_day\ttest_end_day\ttask\n")
                for key in [fixed_key] + rolling_logs:
                    for task in ("buy", "cat", "click", "ext"):
                        handle.write("%s\t%s\t%s\t%s\n" % (key + (task,)))

            report = validate(root, config)
            self.assertTrue(report["ok"], report["errors"])
            self.assertEqual(
                ["20260728"],
                report["artifacts"]["allowed_missing_train_days"])
            self.assertEqual(
                ["20260728"],
                report["artifacts"]["allowed_missing_test_days"])


if __name__ == "__main__":
    unittest.main()
