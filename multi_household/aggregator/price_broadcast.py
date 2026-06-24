"""Aggregator — receives per-household forecasts, computes a dynamic price
signal, broadcasts to all households.

Two DR modes (per timestep):
  • Price-Based: aggregate forecast within capacity. Standard ToU price.
  • Peak Clipping: aggregate exceeds grid threshold G. Apply linear surcharge.
                   Also raises a `peak_event = True` flag that triggers
                   agent-side aggressive deferral.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable

from multi_household.config import (
    PEAK_HOURS_LOCAL, OFFPEAK_HOURS,
    PEAK_PRICE_GBP, MID_PRICE_GBP, OFFPEAK_PRICE_GBP,
    GRID_THRESHOLD_W, PEAK_CLIP_ALPHA,
)


@dataclass
class Broadcast:
    """One coordination message, sent every 10 min to all households."""
    timestep: int
    hour: int
    p_now_gbp_kwh: float        # price for THIS coming step
    p_off_gbp_kwh: float        # price during next off-peak window
    peak_event: bool            # True when aggregator triggers Peak Clipping
    aggregate_forecast_w: float # sum of all household forecasts
    threshold_w: float          # GRID_THRESHOLD_W
    overage_ratio: float        # 0 if Y <= G, else (Y-G)/G
    hold_release: bool = False  # True if aggregator wants houses to pause
                                # off-peak releases this step (natural spike)

    def to_dict(self) -> dict:
        return asdict(self)


def base_tou_price(hour: int) -> float:
    """Standard time-of-use price table."""
    if OFFPEAK_HOURS[0] <= hour < OFFPEAK_HOURS[1]:
        return OFFPEAK_PRICE_GBP
    if PEAK_HOURS_LOCAL[0] <= hour < PEAK_HOURS_LOCAL[1]:
        return PEAK_PRICE_GBP
    return MID_PRICE_GBP


def aggregate_and_price(forecasts_w: Iterable[float],
                        timestep: int,
                        hour: int,
                        grid_threshold_w: float = GRID_THRESHOLD_W,
                        alpha: float = PEAK_CLIP_ALPHA,
                        prev_served_w: float = 0.0,
                        hold_release_threshold_w: float | None = None,
                        ) -> Broadcast:
    """Take per-house forecasts in Watts, return a Broadcast.

    `prev_served_w` is the actual aggregate served at the PREVIOUS timestep.
    If it is near or above the grid threshold (natural spike), aggregator
    raises a hold_release flag so houses don't dump their pool on top of it.
    """
    Y = float(sum(forecasts_w))
    base = base_tou_price(hour)
    p_off = OFFPEAK_PRICE_GBP

    # Natural high-load detection — hold release this step
    if hold_release_threshold_w is None:
        hold_release_threshold_w = 0.85 * grid_threshold_w
    hold_release = prev_served_w >= hold_release_threshold_w

    if Y > grid_threshold_w:
        overage = (Y - grid_threshold_w) / grid_threshold_w
        p_now = base * (1.0 + alpha * overage)
        return Broadcast(
            timestep=timestep, hour=hour,
            p_now_gbp_kwh=round(p_now, 4),
            p_off_gbp_kwh=p_off,
            peak_event=True,
            aggregate_forecast_w=round(Y, 1),
            threshold_w=grid_threshold_w,
            overage_ratio=round(overage, 4),
            hold_release=hold_release,
        )
    return Broadcast(
        timestep=timestep, hour=hour,
        p_now_gbp_kwh=base,
        p_off_gbp_kwh=p_off,
        peak_event=False,
        aggregate_forecast_w=round(Y, 1),
        threshold_w=grid_threshold_w,
        overage_ratio=0.0,
        hold_release=hold_release,
    )


def is_off_peak(hour: int) -> bool:
    return OFFPEAK_HOURS[0] <= hour < OFFPEAK_HOURS[1]
