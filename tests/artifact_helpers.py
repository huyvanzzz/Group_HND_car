"""Validate configurable experiment settings without freezing their chosen values."""
import math


def assert_valid_entropy(value):
    assert type(value) in (int, float) and math.isfinite(value) and value >= 0


def assert_valid_budgets(raw):
    fusion = raw['model']['fusion']
    total = len(raw['data']['view_order']) * raw['model']['vision']['seq_len']
    minimum = len(raw['data']['view_order']) * fusion.get('min_tokens_per_camera', 1)
    stage1, stage2 = fusion['stage1_top_k'], fusion['stage2_top_k']
    assert type(stage1) is int and type(stage2) is int
    assert minimum <= stage2 <= stage1 <= total
    if 'stage3_top_k' in fusion:
        stage3 = fusion['stage3_top_k']
        assert type(stage3) is int and minimum <= stage3 < stage2 < stage1
