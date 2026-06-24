"""Unit tests for the closed-loop wiring (user_choices → agent suppression).

Run:  pytest multi_household/tests/test_closed_loop.py -v
"""
from __future__ import annotations
import json
import pytest
from pathlib import Path

from multi_household.experiments.rollout import load_user_rejection_rates
from multi_household.experiments.metrics import REPORTS


@pytest.fixture
def temp_user_choices(tmp_path, monkeypatch):
    """Create a fake user_choices.json in a temp directory and patch REPORTS
    so load_user_rejection_rates reads from there."""
    fake = tmp_path / "user_choices.json"
    fake.parent.mkdir(parents=True, exist_ok=True)

    # House 7 has rejected washing_machine@mid 3 times, accepted once
    # House 7 has accepted dishwasher@peak once (no rejections)
    # House 9 has only 1 entry (below MIN_SAMPLES, should be ignored)
    log = [
        # House 7 — washing_machine, step 2303 (hour 23 = mid)
        {"house": 7, "rec_step": 2303, "rec_appliance": "washing_machine", "user_choice": 1},
        {"house": 7, "rec_step": 2303, "rec_appliance": "washing_machine", "user_choice": 2},
        {"house": 7, "rec_step": 2303, "rec_appliance": "washing_machine", "user_choice": 2},
        {"house": 7, "rec_step": 2303, "rec_appliance": "washing_machine", "user_choice": 2},
        # House 7 — dishwasher, step 2580 (= day 17, step 144*17+...= step within day 12 → hour 2 = off-peak)
        {"house": 7, "rec_step": 2450, "rec_appliance": "dishwasher", "user_choice": 1},
        {"house": 7, "rec_step": 2456, "rec_appliance": "dishwasher", "user_choice": 1},
        # House 9 — only 1 entry, below MIN_SAMPLES
        {"house": 9, "rec_step": 1000, "rec_appliance": "washing_machine", "user_choice": 2},
    ]
    fake.write_text(json.dumps(log, ensure_ascii=False), encoding="utf-8")

    # monkey patch REPORTS in the rollout module to point at tmp_path
    import multi_household.experiments.rollout as roll
    monkeypatch.setattr(roll, "REPORTS", tmp_path)
    return fake


def test_load_rates_basic(temp_user_choices):
    """House 7 should have 1 pattern with high reject rate."""
    rates = load_user_rejection_rates(7)
    # washing_machine at step 2303 → hour (2303 % 144) // 6 = 143//6 = 23 → mid
    # 3 rejects out of 4 = 0.75
    assert "washing_machine@mid" in rates
    assert rates["washing_machine@mid"] == 0.75


def test_load_rates_below_min_samples_ignored(temp_user_choices):
    """House 9 has 1 entry → ignored (need >= 2)."""
    rates = load_user_rejection_rates(9)
    assert rates == {}                  # nothing returned


def test_load_rates_zero_rejection(temp_user_choices):
    """House 7 dishwasher has 0 rejects → rate = 0.0 (not filtered out)."""
    rates = load_user_rejection_rates(7)
    # Both dishwasher entries are accept (choice=1), so reject rate = 0
    # step 2450 → hour (2450 % 144) // 6 = (98) // 6 = 16 → mid
    # step 2456 → hour (98 + 1) = 98 + 6 = 104 // 6 → wait need to recompute
    # 2456 % 144 = 104, 104 // 6 = 17 → peak. So they're DIFFERENT buckets.
    # Each bucket has only 1 sample → below MIN_SAMPLES → not in result
    # Actually 2450 % 144 = 2450 - 17*144 = 2450-2448 = 2, hour = 0... wait
    # 2450 / 144 = 17.01, so 2450 = 17*144 + 2 → step within day = 2 → hour 0
    # 2456 = 17*144 + 8 → step within day = 8 → hour 1
    # Both are off-peak. 2 entries, 0 rejects → rate = 0
    assert rates.get("dishwasher@off-peak", -1) == 0.0


def test_load_rates_no_house(temp_user_choices):
    """Asking about a house not in the log returns {}."""
    assert load_user_rejection_rates(99) == {}


def test_load_rates_no_file(tmp_path, monkeypatch):
    """If user_choices.json doesn't exist, return {} (no crash)."""
    import multi_household.experiments.rollout as roll
    monkeypatch.setattr(roll, "REPORTS", tmp_path)
    rates = load_user_rejection_rates(7)
    assert rates == {}


def test_load_rates_corrupt_file(tmp_path, monkeypatch):
    """If user_choices.json is malformed, return {} (no crash)."""
    bad = tmp_path / "user_choices.json"
    bad.write_text("{ not valid json", encoding="utf-8")
    import multi_household.experiments.rollout as roll
    monkeypatch.setattr(roll, "REPORTS", tmp_path)
    rates = load_user_rejection_rates(7)
    assert rates == {}
