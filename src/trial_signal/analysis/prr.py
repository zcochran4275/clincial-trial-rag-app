"""Proportional reporting ratio (PRR) — standard FAERS disproportionality measure.

                 event   not event
    drug           a         b
    not drug       c         d

    PRR = [a / (a + b)] / [c / (c + d)]

Common screening rule (Evans et al., 2001): PRR >= 2, chi-square >= 4, and a >= 3.
A "signal" is a hypothesis to investigate, never proof the drug causes the event.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PRRResult:
    a: int
    b: int
    c: int
    d: int
    prr: float
    ci_low: float
    ci_high: float
    chi2: float

    @property
    def is_signal(self) -> bool:
        return self.prr >= 2 and self.chi2 >= 4 and self.a >= 3


def contingency(a: int, drug_total: int, event_total: int, n_total: int) -> tuple[int, int, int, int]:
    b = drug_total - a
    c = event_total - a
    d = n_total - a - b - c
    if min(a, b, c, d) < 0:
        raise ValueError(f"inconsistent counts: a={a} drug_total={drug_total} event_total={event_total} N={n_total}")
    return a, b, c, d


def prr(a: int, drug_total: int, event_total: int, n_total: int) -> PRRResult:
    a, b, c, d = contingency(a, drug_total, event_total, n_total)
    if a == 0 or c == 0:
        raise ValueError("PRR undefined when a or c is zero")
    value = (a / (a + b)) / (c / (c + d))
    se = math.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    lo, hi = (math.exp(math.log(value) + z * se) for z in (-1.96, 1.96))
    # Yates-corrected chi-square for the 2x2 table
    n = a + b + c + d
    num = n * (abs(a * d - b * c) - n / 2) ** 2
    den = (a + b) * (c + d) * (a + c) * (b + d)
    chi2 = num / den if den else 0.0
    return PRRResult(a, b, c, d, value, lo, hi, chi2)
