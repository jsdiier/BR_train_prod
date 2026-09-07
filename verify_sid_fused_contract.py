#!/usr/bin/env python3
import json
import re

import model_conf as conf


def flatten(groups):
    return [slot for group in groups for slot in group]


def main():
    experiment = json.load(open("experiment.json"))
    source = open("model.py").read()
    click_slots = flatten(conf.jz_v3_sid_click_seq_fields)
    pay_slots = flatten(conf.jz_v3_sid_pay_seq_fields)
    candidates = conf.jz_v3_sid_candidate_slots
    native_slots = flatten(conf.jz_v3_native_pay_seq_fields + conf.jz_v3_native_click_seq_fields)

    assert experiment["sid_sequence_mode"] == "aligned_fused_2seq"
    assert experiment["sid_sequence_paths"] == 2
    assert conf.jz_v3_sid_sequence_mode == "aligned_fused_2seq"
    assert len(candidates) == 5 and len(set(candidates)) == 5
    assert len(click_slots) == 250 and len(set(click_slots)) == 250
    assert len(pay_slots) == 250 and len(set(pay_slots)) == 250
    assert all(len(field) == 50 for field in conf.jz_v3_sid_click_seq_fields)
    assert all(len(field) == 50 for field in conf.jz_v3_sid_pay_seq_fields)
    assert set(candidates + click_slots + pay_slots) <= set(conf.all_slot_ids)
    assert len(native_slots) == 600 and len(set(native_slots)) == 600
    assert set(native_slots) <= set(conf.all_slot_ids)
    assert experiment["native_sequence_paths"] == 2
    assert "jz_v3_sid_click_attention_layers" not in source
    assert "jz_v3_sid_pay_attention_layers" not in source
    pattern = (r"DIN_attention_Layer\(\s*\n?\s*\[50, 20\], 'sigmoid', "
               r"name='jz_v3_sid_(?:click|pay)_fused_seq'\)")
    assert len(re.findall(pattern, source)) == 2
    assert "return [click_output, pay_output]" in source

    print("SID_FUSED_CONTRACT_OK")
    print("registered_slots=%d native_paths=2 sid_sequence_paths=2" % len(set(conf.all_slot_ids)))


if __name__ == "__main__":
    main()
