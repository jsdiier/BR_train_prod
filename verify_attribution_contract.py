#!/usr/bin/env python3
"""Static contract for the five-arm JZ-v3 source-attribution batch."""

import json

import model_conf


EXPECTED_DATES = {
    "train_start_day": "20260303",
    "old_train_end_day": "20260817",
    "new_train_start_day": "20260818",
    "train_end_day": "20260825",
    "test_start_day": "20260829",
    "test_end_day": "20260831",
    "auto_test_start_ckpt_day": "20260831",
    "auto_test_end_day": "20260906",
}

NATIVE_RANGES = set(range(33550, 34150))
CANDIDATE_SID_SLOTS = set(range(3013, 3018))
SID_SEQUENCE_RANGES = set(range(341500, 342000))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    with open("experiment.json", encoding="utf-8") as handle:
        experiment = json.load(handle)
    with open("model.py", encoding="utf-8") as handle:
        model_source = handle.read()

    require(
        experiment["baseline"] == "BR_train_EMA_lr_bs_baseline",
        "unexpected batch control",
    )
    require(
        experiment["batch_group"] == "jz_v3_3source_attribution_20260908",
        "unexpected batch group",
    )
    for key, expected in EXPECTED_DATES.items():
        require(str(experiment.get(key)) == expected, "%s mismatch" % key)
    require(experiment.get("expected_metric_rows") == 28, "metric rows must be 28")

    factorial = experiment["factorial"]
    all_slots = set(model_conf.all_slot_ids)
    sparse_slots = set(model_conf.sparse_slot_ids)
    require(
        len(model_conf.all_slot_ids) == len(all_slots),
        "all_slot_ids contains duplicates",
    )

    native_actual = all_slots & NATIVE_RANGES
    native_enabled = bool(factorial["native_shop_sequence"])
    require(
        native_actual == (NATIVE_RANGES if native_enabled else set()),
        "native Shop sequence registration mismatch",
    )
    require(
        ("jz_v3_native_pay_seq_fields" in model_source) == native_enabled,
        "native Shop sequence consumption mismatch",
    )

    candidate_actual = sparse_slots & CANDIDATE_SID_SLOTS
    candidate_enabled = bool(factorial["candidate_shop_sid"])
    require(
        candidate_actual == (CANDIDATE_SID_SLOTS if candidate_enabled else set()),
        "candidate SID registration mismatch",
    )

    sid_actual = all_slots & SID_SEQUENCE_RANGES
    sid_enabled = bool(factorial["sid_sequence"])
    require(
        sid_actual == (SID_SEQUENCE_RANGES if sid_enabled else set()),
        "SID sequence registration mismatch",
    )
    require(
        ("_encode_jz_v3_sid_sequences" in model_source) == sid_enabled,
        "SID sequence consumption mismatch",
    )
    if sid_enabled:
        require(
            "def _encode_jz_v3_sid_sequences(self, pooled_output, slot_mask, emb_shop)"
            in model_source,
            "SID sequence must use emb_shop as DIN query",
        )
        require(experiment.get("sid_sequence_paths") == 2, "SID paths must equal 2")

    print("ATTRIBUTION_CONTRACT_OK branch=%s" % experiment["branch"])
    print(
        "registered native=%d candidate_sid=%d sid_sequence=%d total=%d"
        % (len(native_actual), len(candidate_actual), len(sid_actual), len(all_slots))
    )


if __name__ == "__main__":
    main()
