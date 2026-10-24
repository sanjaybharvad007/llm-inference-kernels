"""Tile-config bookkeeping that does not need a GPU."""
import json
import os

import torch

from fused_attn import ops


def test_simt_config_lookup_uses_seqlen_buckets(tmp_path, monkeypatch):
    table = {"simt": {"bfloat16/128": {"1024": 3, "8192": 0, "65536": 1}}, "triton": {}}
    p = tmp_path / "tuned.json"
    p.write_text(json.dumps(table))
    monkeypatch.setattr(ops, "_TUNED_PATH", str(p))
    monkeypatch.setattr(ops, "_tuned", None)
    assert ops.simt_config_for(torch.bfloat16, 128, 512) == 3
    assert ops.simt_config_for(torch.bfloat16, 128, 4096) == 0
    assert ops.simt_config_for(torch.bfloat16, 128, 100_000) == 1      # beyond the last bucket: largest bucket's choice
    assert ops.simt_config_for(torch.float16, 128, 512) == 0           # untuned dtype -> default


def test_shipped_table_is_valid_json_with_default_sections():
    with open(os.path.join(os.path.dirname(ops.__file__), "tuned_configs.json")) as f:
        t = json.load(f)
    assert set(t) >= {"simt", "triton"}
