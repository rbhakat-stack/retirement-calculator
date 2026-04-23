"""
Unit tests for forecast.calculate_forecast_multi_asset.

The forecast is a monthly simulation with yearly output. Tests target
invariants and observable behavior rather than specific dollar values,
because the monthly compounding makes exact arithmetic brittle.
"""

import pytest
import pandas as pd

from forecast import calculate_forecast_multi_asset


EXPECTED_COLUMNS = {
    "Age", "Is Retired", "Required Spend", "Guaranteed Income",
    "Portfolio Withdrawal", "Cash", "Bonds", "ETFs", "401k", "End Balance",
}


def _base_kwargs(**overrides):
    kwargs = dict(
        current_age=50,
        retire_age=60,
        life_expectancy=95,
        annual_spend_today=100_000,
        inflation_rate=0.03,
        ss_start_age=67,
        social_security_annual_today=30_000,
        annual_contribution=50_000,
        cash_bal=200_000,
        bonds_bal=400_000,
        etfs_bal=400_000,
        k401_bal=200_000,
        cash_yield=0.04,
        bonds_yield=0.05,
        etfs_yield=0.07,
        k401_yield=0.07,
        flow_mode="cash_first",
    )
    kwargs.update(overrides)
    return kwargs


# ---------- shape & column contract ----------

def test_returns_dataframe_with_expected_columns():
    df = calculate_forecast_multi_asset(**_base_kwargs())
    assert isinstance(df, pd.DataFrame)
    assert EXPECTED_COLUMNS.issubset(df.columns)


def test_first_row_is_current_age():
    df = calculate_forecast_multi_asset(**_base_kwargs(current_age=45))
    assert int(df.iloc[0]["Age"]) == 45


def test_not_empty_for_normal_inputs():
    df = calculate_forecast_multi_asset(**_base_kwargs())
    assert len(df) > 1


# ---------- horizon behavior ----------

def test_zero_horizon_when_current_age_equals_life_expectancy():
    # total_months = 0 → only the initial record is produced
    df = calculate_forecast_multi_asset(
        **_base_kwargs(current_age=95, retire_age=95, life_expectancy=95, ss_start_age=95)
    )
    assert len(df) == 1


def test_horizon_length_no_depletion_case():
    # Strong portfolio, small withdrawal — should reach life_expectancy
    df = calculate_forecast_multi_asset(
        **_base_kwargs(
            current_age=50, retire_age=60, life_expectancy=70,
            annual_spend_today=10_000,
            cash_bal=1_000_000, bonds_bal=1_000_000,
            etfs_bal=1_000_000, k401_bal=1_000_000,
        )
    )
    # Initial row + 20 annual records
    assert int(df.iloc[-1]["Age"]) == 70


# ---------- growth invariants ----------

def test_working_years_grow_with_contributions_no_withdrawals():
    # Pre-retirement: contributions + growth, no withdrawals → balance must strictly increase
    df = calculate_forecast_multi_asset(
        **_base_kwargs(
            current_age=50, retire_age=60, life_expectancy=60,
            annual_spend_today=0,
            social_security_annual_today=0,
        )
    )
    balances = df["End Balance"].tolist()
    assert balances[-1] > balances[0]


def test_zero_yield_zero_contribution_zero_spend_preserves_balance():
    df = calculate_forecast_multi_asset(
        **_base_kwargs(
            current_age=50, retire_age=50, life_expectancy=52,
            annual_spend_today=0,
            social_security_annual_today=0,
            annual_contribution=0,
            cash_yield=0.0, bonds_yield=0.0, etfs_yield=0.0, k401_yield=0.0,
            inflation_rate=0.0,
        )
    )
    # Already-retired with zero spend and zero growth — balance is flat
    initial = df.iloc[0]["End Balance"]
    final = df.iloc[-1]["End Balance"]
    assert final == pytest.approx(initial)


# ---------- depletion ----------

def test_depletion_breaks_simulation_early():
    # Tiny portfolio, already retired, large spend → must deplete
    # The simulation breaks as soon as total_pool() <= 0. If depletion lands
    # mid-year, the final yearly record may be the last captured year-end
    # snapshot before the pool hit zero — what matters is the horizon stops
    # well short of life_expectancy.
    df = calculate_forecast_multi_asset(
        **_base_kwargs(
            current_age=65, retire_age=65, life_expectancy=95,
            annual_spend_today=200_000,
            social_security_annual_today=0,
            annual_contribution=0,
            cash_bal=50_000, bonds_bal=0, etfs_bal=0, k401_bal=0,
            cash_yield=0.0, bonds_yield=0.0, etfs_yield=0.0, k401_yield=0.0,
        )
    )
    # Simulation must terminate well before reaching life_expectancy
    assert int(df.iloc[-1]["Age"]) < 95


def test_depletion_shrinks_balance_and_truncates_horizon():
    # Depletion scenario: each captured year-end balance must be strictly
    # decreasing, and the simulation must terminate short of life_expectancy.
    df = calculate_forecast_multi_asset(
        **_base_kwargs(
            current_age=65, retire_age=65, life_expectancy=95,
            annual_spend_today=100_000,
            social_security_annual_today=0,
            annual_contribution=0,
            cash_bal=500_000, bonds_bal=0, etfs_bal=0, k401_bal=0,
            cash_yield=0.0, bonds_yield=0.0, etfs_yield=0.0, k401_yield=0.0,
        )
    )
    balances = df["End Balance"].tolist()
    # Strictly decreasing — no year's end balance is larger than the previous
    for prev, curr in zip(balances, balances[1:]):
        assert curr < prev
    # Final captured balance is materially smaller than starting balance
    assert balances[-1] < balances[0] * 0.25
    # Horizon truncated by depletion
    assert int(df.iloc[-1]["Age"]) < 95


def test_end_balance_never_negative():
    df = calculate_forecast_multi_asset(
        **_base_kwargs(
            current_age=65, retire_age=65, life_expectancy=95,
            annual_spend_today=150_000,
            cash_bal=100_000, bonds_bal=0, etfs_bal=0, k401_bal=0,
        )
    )
    assert (df["End Balance"] >= 0.0).all()


# ---------- withdrawal order (flow_mode) ----------

def test_cash_first_drains_cash_before_other_buckets():
    df = calculate_forecast_multi_asset(
        **_base_kwargs(
            current_age=65, retire_age=65, life_expectancy=70,
            annual_spend_today=150_000,
            social_security_annual_today=0,
            annual_contribution=0,
            cash_bal=100_000, bonds_bal=500_000, etfs_bal=500_000, k401_bal=500_000,
            cash_yield=0.0, bonds_yield=0.0, etfs_yield=0.0, k401_yield=0.0,
            flow_mode="cash_first",
        )
    )
    # After first retired year, cash should be exhausted while other buckets are largely intact
    year1 = df.iloc[1]
    assert year1["Cash"] == pytest.approx(0.0, abs=1.0)
    assert year1["Bonds"] > 400_000  # barely touched
    assert year1["ETFs"] > 400_000
    assert year1["401k"] > 400_000


def test_pro_rata_draws_proportionally_from_all_buckets():
    df = calculate_forecast_multi_asset(
        **_base_kwargs(
            current_age=65, retire_age=65, life_expectancy=66,
            annual_spend_today=120_000,
            social_security_annual_today=0,
            annual_contribution=0,
            cash_bal=250_000, bonds_bal=250_000, etfs_bal=250_000, k401_bal=250_000,
            cash_yield=0.0, bonds_yield=0.0, etfs_yield=0.0, k401_yield=0.0,
            flow_mode="pro_rata",
        )
    )
    year1 = df.iloc[1]
    # All four buckets started equal; after pro-rata draw they should still be roughly equal
    vals = [year1["Cash"], year1["Bonds"], year1["ETFs"], year1["401k"]]
    assert max(vals) - min(vals) < 1.0  # all equal within floating-point noise


# ---------- social security ----------

def test_social_security_zero_before_start_age():
    df = calculate_forecast_multi_asset(**_base_kwargs(ss_start_age=70))
    # Rows where Age < 70 should have zero guaranteed income
    pre_ss = df[df["Age"] < 70]
    assert (pre_ss["Guaranteed Income"] == 0.0).all()


def test_social_security_nonzero_from_start_age():
    df = calculate_forecast_multi_asset(
        **_base_kwargs(ss_start_age=67, social_security_annual_today=30_000)
    )
    at_or_after = df[df["Age"] >= 67]
    if not at_or_after.empty:
        assert (at_or_after["Guaranteed Income"] > 0.0).all()


# ---------- input resilience ----------

def test_negative_bucket_inputs_are_floored_to_zero():
    df = calculate_forecast_multi_asset(
        **_base_kwargs(
            cash_bal=-1000, bonds_bal=-5000, etfs_bal=-500, k401_bal=-100,
            annual_contribution=0, annual_spend_today=0,
            social_security_annual_today=0,
            cash_yield=0.0, bonds_yield=0.0, etfs_yield=0.0, k401_yield=0.0,
        )
    )
    # Must not crash; initial balances floored to 0
    assert df.iloc[0]["Cash"] == 0.0
    assert df.iloc[0]["Bonds"] == 0.0
    assert df.iloc[0]["ETFs"] == 0.0
    assert df.iloc[0]["401k"] == 0.0


def test_is_retired_flag_flips_at_retire_age():
    df = calculate_forecast_multi_asset(**_base_kwargs(current_age=50, retire_age=60))
    pre = df[df["Age"] < 60]
    post = df[df["Age"] >= 60]
    assert (pre["Is Retired"] == False).all()
    if not post.empty:
        assert (post["Is Retired"] == True).all()
