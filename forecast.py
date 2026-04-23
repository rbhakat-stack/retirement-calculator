"""
Pure multi-asset retirement forecast logic.

No Streamlit dependency — safe to import in tests.
Returns a pandas DataFrame with yearly rows derived from a monthly simulation.
"""

import math
import pandas as pd


def calculate_forecast_multi_asset(
    current_age: int,
    retire_age: int,
    life_expectancy: int,
    annual_spend_today: float,
    inflation_rate: float,
    ss_start_age: int,
    social_security_annual_today: float,
    annual_contribution: float,
    cash_bal: float,
    bonds_bal: float,
    etfs_bal: float,
    k401_bal: float,
    cash_yield: float,
    bonds_yield: float,
    etfs_yield: float,
    k401_yield: float,
    flow_mode: str = "cash_first",
):
    """
    Monthly simulation:
    - Expenses inflate monthly
    - SS inflates monthly once started
    - Growth applied monthly per bucket
    - While working: contribution added monthly (allocated based on flow_mode)
    - In retirement: withdrawals funded from buckets (flow_mode controls order)
    Returns a YEARLY DataFrame.
    """
    max_age          = life_expectancy
    total_months     = max(0, (max_age - current_age) * 12)
    retirement_month = max(0, (retire_age - current_age) * 12)

    m_infl  = inflation_rate / 12.0
    m_cash  = cash_yield    / 12.0
    m_bonds = bonds_yield   / 12.0
    m_etfs  = etfs_yield    / 12.0
    m_k401  = k401_yield    / 12.0

    m_spend   = annual_spend_today             / 12.0
    m_ss      = social_security_annual_today   / 12.0
    m_contrib = annual_contribution            / 12.0

    cash  = float(max(0, cash_bal))
    bonds = float(max(0, bonds_bal))
    etfs  = float(max(0, etfs_bal))
    k401  = float(max(0, k401_bal))

    def total_pool():
        return cash + bonds + etfs + k401

    def allocate_surplus(amount: float):
        nonlocal cash, bonds, etfs, k401
        if amount <= 0:
            return
        pool = total_pool()
        if pool <= 0:
            add = amount / 4.0
            cash += add; bonds += add; etfs += add; k401 += add
            return
        if flow_mode == "pro_rata":
            cash  += amount * (cash  / pool) if cash  > 0 else 0
            bonds += amount * (bonds / pool) if bonds > 0 else 0
            etfs  += amount * (etfs  / pool) if etfs  > 0 else 0
            k401  += amount * (k401  / pool) if k401  > 0 else 0
        else:
            cash += amount

    def withdraw_deficit(amount: float):
        nonlocal cash, bonds, etfs, k401
        if amount <= 0:
            return
        if flow_mode == "cash_first":
            if amount > 0 and cash > 0:
                take = min(amount, cash)
                cash -= take
                amount -= take
            if amount > 0 and bonds > 0:
                take = min(amount, bonds)
                bonds -= take
                amount -= take
            if amount > 0 and etfs > 0:
                take = min(amount, etfs)
                etfs -= take
                amount -= take
            if amount > 0 and k401 > 0:
                take = min(amount, k401)
                k401 -= take
                amount -= take
        else:
            pool = total_pool()
            if pool <= 0:
                return
            w     = min(amount, pool)
            ratio = w / pool
            cash  -= cash  * ratio
            bonds -= bonds * ratio
            etfs  -= etfs  * ratio
            k401  -= k401  * ratio

        cash  = max(0.0, cash)
        bonds = max(0.0, bonds)
        etfs  = max(0.0, etfs)
        k401  = max(0.0, k401)

    col_age                  = []
    col_is_retired           = []
    col_required_spend       = []
    col_guaranteed_income    = []
    col_portfolio_withdrawal = []
    col_cash                 = []
    col_bonds                = []
    col_etfs                 = []
    col_k401                 = []
    col_end_balance          = []

    def _record(age_val, is_ret, req_sp, guar_inc, port_with, c, b, e, k, total):
        col_age.append(age_val)
        col_is_retired.append(is_ret)
        col_required_spend.append(req_sp)
        col_guaranteed_income.append(guar_inc)
        col_portfolio_withdrawal.append(port_with)
        col_cash.append(c)
        col_bonds.append(b)
        col_etfs.append(e)
        col_k401.append(k)
        col_end_balance.append(total)

    _record(
        current_age,
        current_age >= retire_age,
        annual_spend_today,
        0.0 if current_age < ss_start_age else social_security_annual_today,
        0.0,
        cash, bonds, etfs, k401,
        total_pool(),
    )

    for month in range(1, total_months + 1):
        sim_age    = current_age + month / 12.0
        age_int    = int(math.floor(sim_age))
        is_retired = month >= retirement_month

        m_spend *= (1.0 + m_infl)
        m_ss    *= (1.0 + m_infl)

        cash  *= (1.0 + m_cash)
        bonds *= (1.0 + m_bonds)
        etfs  *= (1.0 + m_etfs)
        k401  *= (1.0 + m_k401)

        guaranteed_month = m_ss if sim_age >= ss_start_age else 0.0

        if not is_retired and m_contrib > 0:
            allocate_surplus(m_contrib)

        monthly_need = 0.0
        if is_retired:
            monthly_need = max(0.0, m_spend - guaranteed_month)
            withdraw_deficit(monthly_need)

        if month % 12 == 0:
            _record(
                age_int,
                age_int >= retire_age,
                m_spend * 12.0,
                guaranteed_month * 12.0,
                monthly_need * 12.0,
                cash, bonds, etfs, k401,
                total_pool(),
            )

        if total_pool() <= 0:
            break

    return pd.DataFrame({
        "Age":                  col_age,
        "Is Retired":           col_is_retired,
        "Required Spend":       col_required_spend,
        "Guaranteed Income":    col_guaranteed_income,
        "Portfolio Withdrawal": col_portfolio_withdrawal,
        "Cash":                 col_cash,
        "Bonds":                col_bonds,
        "ETFs":                 col_etfs,
        "401k":                 col_k401,
        "End Balance":          col_end_balance,
    })
