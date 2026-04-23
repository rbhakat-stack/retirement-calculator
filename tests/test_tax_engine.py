"""
Unit tests for tax_engine.py.

Covers progressive-tax arithmetic, federal/state/credit interactions,
bracket boundaries, and edge cases (zero income, below-deduction income,
credit phase-out at exact boundaries).
"""

import math
import pytest

from tax_engine import (
    FEDERAL_BRACKETS,
    NJ_BRACKETS,
    calculate_progressive_tax,
    calculate_annual_taxes,
)


# ---------- calculate_progressive_tax ----------

def test_progressive_tax_zero_income_is_zero():
    assert calculate_progressive_tax(0.0, FEDERAL_BRACKETS["single"]) == 0.0


def test_progressive_tax_within_first_bracket():
    # Single, $10,000 taxable → 10% flat
    assert calculate_progressive_tax(10_000, FEDERAL_BRACKETS["single"]) == pytest.approx(1_000.0)


def test_progressive_tax_at_first_bracket_boundary():
    # Exactly at 11,600 → 10% on full amount
    assert calculate_progressive_tax(11_600, FEDERAL_BRACKETS["single"]) == pytest.approx(1_160.0)


def test_progressive_tax_spans_two_brackets():
    # 20,000: first 11,600 @ 10% + 8,400 @ 12% = 1,160 + 1,008 = 2,168
    assert calculate_progressive_tax(20_000, FEDERAL_BRACKETS["single"]) == pytest.approx(2_168.0)


def test_progressive_tax_spans_multiple_brackets_single_100k():
    # 11,600*.10 + (47,150-11,600)*.12 + (100,000-47,150)*.22
    expected = 1_160 + 35_550 * 0.12 + 52_850 * 0.22
    assert calculate_progressive_tax(100_000, FEDERAL_BRACKETS["single"]) == pytest.approx(expected)


def test_progressive_tax_into_top_bracket():
    # 1M single uses the top 37% bracket for the last slice
    result = calculate_progressive_tax(1_000_000, FEDERAL_BRACKETS["single"])
    # Sanity: must be strictly less than 37% flat
    assert 0 < result < 1_000_000 * 0.37


# ---------- calculate_annual_taxes: federal standard deduction ----------

def test_income_below_single_deduction_yields_no_federal_tax():
    r = calculate_annual_taxes(
        gross_income=10_000, status="single", state_code="Other",
        manual_state_rate=0.0, dependents=0,
    )
    assert r["federal"] == 0.0


def test_income_below_married_deduction_yields_no_federal_tax():
    r = calculate_annual_taxes(
        gross_income=25_000, status="married", state_code="Other",
        manual_state_rate=0.0, dependents=0,
    )
    assert r["federal"] == 0.0


def test_single_vs_married_deduction_difference():
    # Same gross income, married should produce lower federal tax due to higher deduction + wider brackets
    single = calculate_annual_taxes(50_000, "single", "Other", 0.0, 0)
    married = calculate_annual_taxes(50_000, "married", "Other", 0.0, 0)
    assert married["federal"] < single["federal"]


# ---------- calculate_annual_taxes: state ----------

def test_other_state_is_flat_percentage_of_gross():
    r = calculate_annual_taxes(
        gross_income=100_000, status="single", state_code="Other",
        manual_state_rate=5.0, dependents=0,
    )
    assert r["state"] == pytest.approx(5_000.0)


def test_nj_state_tax_uses_progressive_brackets():
    # NJ single, 50k gross, 0 deps
    # exemption = 1,000 → taxable 49,000
    # 20,000*.014 + 15,000*.0175 + 5,000*.035 + 9,000*.05525
    # = 280 + 262.5 + 175 + 497.25 = 1,214.75
    r = calculate_annual_taxes(50_000, "single", "NJ", 0.0, 0)
    assert r["state"] == pytest.approx(1_214.75)


def test_nj_state_tax_married_has_different_brackets():
    # NJ married vs single at same income differ due to bracket boundaries and exemption
    single = calculate_annual_taxes(80_000, "single", "NJ", 0.0, 0)
    married = calculate_annual_taxes(80_000, "married", "NJ", 0.0, 0)
    assert single["state"] != married["state"]


# ---------- calculate_annual_taxes: child credit ----------

def test_child_credit_full_when_below_phase_out():
    r = calculate_annual_taxes(150_000, "single", "Other", 0.0, dependents=2)
    assert r["credits"] == pytest.approx(4_000.0)


def test_child_credit_unchanged_exactly_at_phase_out_boundary_single():
    # Boundary at 200,000 — code uses strict > so at exactly 200k no reduction
    r = calculate_annual_taxes(200_000, "single", "Other", 0.0, dependents=2)
    assert r["credits"] == pytest.approx(4_000.0)


def test_child_credit_reduced_just_above_phase_out_boundary_single():
    # 200,001 → ceil(1/1000)=1 → reduction = 50
    r = calculate_annual_taxes(200_001, "single", "Other", 0.0, dependents=2)
    assert r["credits"] == pytest.approx(3_950.0)


def test_child_credit_phase_out_married_uses_400k_threshold():
    # At 400,001 married: ceil(1/1000)=1 → reduction 50
    r = calculate_annual_taxes(400_001, "married", "Other", 0.0, dependents=2)
    assert r["credits"] == pytest.approx(3_950.0)


def test_child_credit_fully_phased_out_floor_at_zero():
    # Very high income → reduction exceeds credit, floored at 0
    r = calculate_annual_taxes(10_000_000, "single", "Other", 0.0, dependents=2)
    assert r["credits"] == 0.0


def test_no_dependents_means_zero_credits():
    r = calculate_annual_taxes(200_000, "single", "Other", 0.0, dependents=0)
    assert r["credits"] == 0.0


def test_credit_reduces_federal_tax_not_below_zero():
    # Very low income with many dependents → federal_tax would go negative without the floor
    r = calculate_annual_taxes(30_000, "single", "Other", 0.0, dependents=10)
    assert r["federal"] >= 0.0


# ---------- calculate_annual_taxes: aggregation ----------

def test_total_equals_federal_plus_state():
    r = calculate_annual_taxes(250_000, "married", "NJ", 0.0, dependents=1)
    assert r["total"] == pytest.approx(r["federal"] + r["state"])


def test_effective_rate_matches_total_over_gross():
    r = calculate_annual_taxes(250_000, "married", "NJ", 0.0, 0)
    assert r["effective_rate"] == pytest.approx(r["total"] / 250_000)


def test_zero_gross_income_yields_zero_effective_rate():
    r = calculate_annual_taxes(0, "single", "NJ", 0.0, 0)
    assert r["effective_rate"] == 0.0
    assert r["total"] == 0.0


def test_return_shape_has_all_expected_keys():
    r = calculate_annual_taxes(100_000, "single", "Other", 5.0, 1)
    assert set(r.keys()) == {"federal", "state", "credits", "total", "effective_rate"}
