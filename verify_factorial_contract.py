#!/usr/bin/env python3
import json

import model_conf as conf


def flatten(groups):
    return [slot for group in groups for slot in group]


def main():
    experiment = json.load(open("experiment.json"))
    expected_native = bool(experiment["factorial"]["native_sequence"])
    expected_sid = bool(experiment["factorial"]["sid"])
    assert conf.enable_jz_v3_native_seq == expected_native
    assert conf.enable_jz_v3_sid == expected_sid

    native = set(flatten(
        conf.jz_v3_native_pay_seq_fields + conf.jz_v3_native_click_seq_fields))
    sid_sequence = set(flatten(
        conf.jz_v3_sid_click_seq_fields + conf.jz_v3_sid_pay_seq_fields))
    sid_candidate = set(conf.jz_v3_sid_candidate_slots)
    registered = set(conf.all_slot_ids)

    assert len(native) == 600
    assert len(sid_sequence) == 500
    assert len(sid_candidate) == 5
    assert not (native & sid_sequence)
    assert not (native & sid_candidate)
    assert not (sid_sequence & sid_candidate)
    assert (native <= registered) == expected_native
    assert (sid_sequence <= registered) == expected_sid
    assert (sid_candidate <= registered) == expected_sid
    assert (sid_candidate <= set(conf.sparse_slot_ids)) == expected_sid
    assert (sid_candidate <= set(conf.lr_slot_ids)) == expected_sid
    assert (sid_candidate <= set(conf.shop_fea_list)) == expected_sid

    assert set(range(32841, 32861)) <= registered
    assert set(range(32861, 32881)) <= registered
    assert len(conf.all_slot_ids) == len(registered)
    expected_registered_count = 1544 + (600 if expected_native else 0) + (505 if expected_sid else 0)
    assert len(registered) == expected_registered_count, (
        len(registered), expected_registered_count)

    print("FACTORIAL_CONTRACT_OK")
    print("native_sequence=%s sid=%s registered_slots=%d" % (
        expected_native, expected_sid, len(conf.all_slot_ids)))


if __name__ == "__main__":
    main()
