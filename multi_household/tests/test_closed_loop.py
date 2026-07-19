"""Unit tests for the closed-loop wiring (user_choices → agent suppression).

These pin the v2 provenance rules:
  • the hour comes ONLY from the stored `rec_hour` (the old code reconstructed
    it as (step % 144)//6, which assumed the test window starts at midnight —
    it starts at noon/09:40, so every reconstruction disagreed with the truth);
  • duplicate clicks on one recommendation_id count ONCE (last click wins);
  • legacy entries without rec_hour, and non-"real" sources, are skipped.

Run:  pytest multi_household/tests/test_closed_loop.py -v
"""
from __future__ import annotations
import json
import pytest

from multi_household.experiments.rollout import load_user_rejection_rates


def _write(tmp_path, entries):
    (tmp_path / "user_choices.json").write_text(
        json.dumps(entries, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def patched_reports(tmp_path, monkeypatch):
    import multi_household.experiments.rollout as roll
    monkeypatch.setattr(roll, "REPORTS", tmp_path)
    return tmp_path


def test_hour_comes_from_stored_field_not_step_arithmetic(patched_reports):
    """rec_hour=18 (peak) must land in the peak bucket even though the step
    arithmetic (step%144)//6 would map step 2303 to hour 23 (mid)."""
    _write(patched_reports, [
        {"house": 7, "recommendation_id": "a", "rec_step": 2303,
         "rec_hour": 18, "rec_appliance": "washing_machine",
         "user_choice": 2, "source": "real"},
        {"house": 7, "recommendation_id": "b", "rec_step": 2450,
         "rec_hour": 18, "rec_appliance": "washing_machine",
         "user_choice": 2, "source": "real"},
    ])
    rates = load_user_rejection_rates(7)
    assert rates == {"washing_machine@peak": 1.0}


def test_duplicate_clicks_on_one_rec_count_once(patched_reports):
    """The legacy log held 5 clicks on ONE recommendation and treated them as
    5 samples. v2: one rec = one sample, last click wins."""
    clicks = [1, 2, 3, 2, 2]                     # accept, reject, modify, ...
    _write(patched_reports, [
        {"house": 7, "recommendation_id": "same-rec", "rec_step": 100,
         "rec_hour": 2, "rec_appliance": "washing_machine",
         "user_choice": c, "source": "real"} for c in clicks
    ] + [
        # a second, distinct rejected rec so the pattern reaches MIN_SAMPLES=2
        {"house": 7, "recommendation_id": "other-rec", "rec_step": 200,
         "rec_hour": 3, "rec_appliance": "washing_machine",
         "user_choice": 2, "source": "real"},
    ])
    rates = load_user_rejection_rates(7)
    # dedup → 2 samples in off-peak bucket; last click of same-rec is 2 (reject)
    assert rates == {"washing_machine@off-peak": 1.0}


def test_legacy_entries_without_hour_are_skipped(patched_reports):
    _write(patched_reports, [
        {"house": 7, "rec_step": 2303, "rec_appliance": "washing_machine",
         "user_choice": 2},                       # legacy: no rec_hour
        {"house": 7, "rec_step": 2303, "rec_appliance": "washing_machine",
         "user_choice": 2},
    ])
    assert load_user_rejection_rates(7) == {}


def test_non_real_sources_are_skipped(patched_reports):
    _write(patched_reports, [
        {"house": 7, "recommendation_id": f"d{i}", "rec_step": 100 + i,
         "rec_hour": 18, "rec_appliance": "washing_machine",
         "user_choice": 2, "source": "demo-auto"} for i in range(4)
    ])
    assert load_user_rejection_rates(7) == {}


def test_below_min_samples_ignored(patched_reports):
    _write(patched_reports, [
        {"house": 9, "recommendation_id": "x", "rec_step": 10, "rec_hour": 18,
         "rec_appliance": "washing_machine", "user_choice": 2, "source": "real"},
    ])
    assert load_user_rejection_rates(9) == {}


def test_no_house_no_file_corrupt_file(patched_reports, tmp_path):
    assert load_user_rejection_rates(99) == {}          # no entries
    (patched_reports / "user_choices.json").unlink(missing_ok=True)
    assert load_user_rejection_rates(7) == {}           # no file
    (patched_reports / "user_choices.json").write_text("{ not valid json",
                                                       encoding="utf-8")
    assert load_user_rejection_rates(7) == {}           # corrupt file
