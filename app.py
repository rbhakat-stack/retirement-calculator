import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import hmac
import copy
import uuid
import json


# =============================================================================
# Helpers: rate normalization (store rates/yields as decimals internally)
# =============================================================================
def _as_decimal_rate(x, default=0.0) -> float:
    """
    Normalize any rate to decimal form.
    Accepts:
      - 0.07 (already decimal)
      - 7 or 7.0 (percent)  -> 0.07
    """
    try:
        v = float(x)
    except Exception:
        return float(default)
    return v / 100.0 if v > 1.0 else v


def _as_percent_display(x, default_pct=0.0) -> float:
    """
    For slider default display (in percent units).
    Accepts decimal or percent and returns percent number (e.g., 7.0).
    """
    d = _as_decimal_rate(x, default=default_pct / 100.0)
    return d * 100.0


def normalize_snapshot(s: dict) -> dict:
    """
    Returns a NEW dict with consistent units and required keys.
    All rates/yields stored as decimals.
    """
    s2 = copy.deepcopy(s)

    # Core rates
    s2["inflation_rate"] = _as_decimal_rate(s2.get("inflation_rate", 0.03), 0.03)
    s2["pre_retire_return"] = _as_decimal_rate(s2.get("pre_retire_return", 0.07), 0.07)
    s2["post_retire_return"] = _as_decimal_rate(s2.get("post_retire_return", 0.045), 0.045)

    # Multi-asset yields
    s2["cash_yield"] = _as_decimal_rate(s2.get("cash_yield", 0.04), 0.04)
    s2["bonds_yield"] = _as_decimal_rate(s2.get("bonds_yield", 0.05), 0.05)
    s2["etfs_yield"] = _as_decimal_rate(s2.get("etfs_yield", 0.07), 0.07)
    s2["k401_yield"] = _as_decimal_rate(s2.get("k401_yield", 0.07), 0.07)

    # Defaults / required keys
    s2["use_multi_asset"] = bool(s2.get("use_multi_asset", True))
    s2["flow_mode"] = s2.get("flow_mode", "cash_first")
    if s2["flow_mode"] not in ("cash_first", "pro_rata"):
        s2["flow_mode"] = "cash_first"

    # Ensure numeric types for critical fields (avoid strings)
    for k in ["current_age", "retire_age", "life_expectancy", "ss_start_age", "dependents"]:
        if k in s2 and s2[k] is not None:
            s2[k] = int(s2[k])

    for k in [
        "annual_spend_retirement",
        "social_security",
        "annual_contribution",
        "current_portfolio",
        "cash_bal",
        "bonds_bal",
        "etfs_bal",
        "k401_bal",
        "annual_gross_income",
        "manual_state_rate",
        "annual_expenses",
    ]:
        if k in s2 and s2[k] is not None:
            s2[k] = float(s2[k])

    return s2


# =============================================================================
# APP CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Strategic Retirement Planner",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SIMPLE LOGIN GATE (USERID/PASSWORD)
# =============================================================================
def _hash_password(password: str, salt: str) -> str:
    """
    PBKDF2 hash (basic gating). Store only the hash in st.secrets.
    """
    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    )
    return dk.hex()


def require_login():
    if st.session_state.get("is_authenticated", False):
        return

    st.title("Login Required")
    st.caption("Enter your credentials to access the application.")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("User ID")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        allowed_users = st.secrets.get("auth", {}).get("users", {})
        salt = st.secrets.get("auth", {}).get("salt", "")

        if not salt or not allowed_users:
            st.error("Auth is not configured. Please add [auth] secrets (salt + users).")
            st.stop()

        expected_hash = allowed_users.get(username)
        if not expected_hash:
            st.error("Invalid User ID or Password.")
            st.stop()

        computed_hash = _hash_password(password, salt)

        if hmac.compare_digest(computed_hash, expected_hash):
            st.session_state.is_authenticated = True
            st.session_state.current_user = username  # used for per-user state keys
            st.success("Login successful.")
            st.rerun()
        else:
            st.error("Invalid User ID or Password.")
            st.stop()

    st.stop()


require_login()

with st.sidebar:
    if st.button("Log out"):
        st.session_state.is_authenticated = False
        st.session_state.current_user = None
        st.rerun()

# =============================================================================
# GLOBAL STYLE OVERRIDES (incl. legal badge)
# =============================================================================
st.markdown(
    """
    <style>
    /* =========================
       MAIN CONTENT (NARROW)
       ========================= */
    .main .block-container {
        max-width: 960px;
        padding-left: 2rem;
        padding-right: 2rem;
        margin-left: auto;
        margin-right: auto;
    }

    @media (min-width: 1200px) {
        .main .block-container { max-width: 960px; }
    }

    /* =========================
       SIDEBAR (WIDER + CLEANER)
       ========================= */
    section[data-testid="stSidebar"] {
        width: 420px !important;
        min-width: 420px !important;
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }

    section[data-testid="stSidebar"] > div {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    [data-testid="stSidebar"] {
        font-size: 0.9rem !important;
    }

    /* =========================
       TYPOGRAPHY
       ========================= */
    h1 { font-size: 1.6rem !important; font-weight: 600 !important; }
    h2 { font-size: 1.25rem !important; margin-top: 1.2rem !important; margin-bottom: 0.4rem !important; }
    h3 { font-size: 1.05rem !important; margin-top: 0.8rem !important; }

    [data-testid="stMetricValue"] { font-size: 1.3rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: #6B7280 !important; }

    .stMarkdown p { font-size: 0.9rem; line-height: 1.5; }

    /* =========================
       TOP-RIGHT LEGAL BADGE
       ========================= */
    div[data-testid="stAppViewContainer"]::before {
        content: "© 2026 Ranabir Bhakat™ · Proprietary & Confidential · Unauthorized use prohibited";
        position: fixed;
        top: 22px;           /* move down */
        right: 240px;        /* move left */
        z-index: 999999;
        padding: 6px 10px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.2px;
        color: rgba(255, 255, 255, 0.92);
        background: rgba(0, 0, 0, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.18);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        pointer-events: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# TAX LOGIC
# =============================================================================
FEDERAL_BRACKETS = {
    "single": [
        {"limit": 11_600, "rate": 0.10},
        {"limit": 47_150, "rate": 0.12},
        {"limit": 100_525, "rate": 0.22},
        {"limit": 191_950, "rate": 0.24},
        {"limit": 243_725, "rate": 0.32},
        {"limit": 609_350, "rate": 0.35},
        {"limit": float("inf"), "rate": 0.37},
    ],
    "married": [
        {"limit": 23_200, "rate": 0.10},
        {"limit": 94_300, "rate": 0.12},
        {"limit": 201_050, "rate": 0.22},
        {"limit": 383_900, "rate": 0.24},
        {"limit": 487_450, "rate": 0.32},
        {"limit": 731_200, "rate": 0.35},
        {"limit": float("inf"), "rate": 0.37},
    ],
}

NJ_BRACKETS = {
    "single": [
        {"limit": 20_000, "rate": 0.014},
        {"limit": 35_000, "rate": 0.0175},
        {"limit": 40_000, "rate": 0.035},
        {"limit": 75_000, "rate": 0.05525},
        {"limit": 500_000, "rate": 0.0637},
        {"limit": 1_000_000, "rate": 0.0897},
        {"limit": float("inf"), "rate": 0.1075},
    ],
    "married": [
        {"limit": 20_000, "rate": 0.014},
        {"limit": 50_000, "rate": 0.0175},
        {"limit": 70_000, "rate": 0.0245},
        {"limit": 80_000, "rate": 0.035},
        {"limit": 150_000, "rate": 0.05525},
        {"limit": 500_000, "rate": 0.0637},
        {"limit": 1_000_000, "rate": 0.0897},
        {"limit": float("inf"), "rate": 0.1075},
    ],
}


def calculate_progressive_tax(taxable_income: float, brackets) -> float:
    tax = 0.0
    previous_limit = 0.0
    for bracket in brackets:
        limit = bracket["limit"]
        rate = bracket["rate"]
        if taxable_income > previous_limit:
            taxable_amount = min(taxable_income, limit) - previous_limit
            tax += taxable_amount * rate
            previous_limit = limit
        else:
            break
    return tax


def calculate_annual_taxes(
    gross_income: float,
    status: str,
    state_code: str,
    manual_state_rate: float,
    dependents: int = 0,
):
    # Federal standard deduction (2024 approximation)
    standard_deduction = 14_600 if status == "single" else 29_200
    federal_taxable_income = max(0.0, gross_income - standard_deduction)

    federal_tax = calculate_progressive_tax(federal_taxable_income, FEDERAL_BRACKETS[status])

    # Child tax credit (approximate)
    credit_phase_out_start = 400_000 if status == "married" else 200_000
    total_credit = dependents * 2_000

    if gross_income > credit_phase_out_start:
        reduction = np.ceil((gross_income - credit_phase_out_start) / 1_000) * 50
        total_credit = max(0.0, total_credit - reduction)

    federal_tax = max(0.0, federal_tax - total_credit)

    # State tax
    if state_code == "NJ":
        nj_exempt = (dependents * 1_500) + (2_000 if status == "married" else 1_000)
        nj_taxable = max(0.0, gross_income - nj_exempt)
        state_tax = calculate_progressive_tax(nj_taxable, NJ_BRACKETS[status])
    else:
        state_tax = gross_income * (manual_state_rate / 100.0)

    total_tax = federal_tax + state_tax
    effective_rate = total_tax / gross_income if gross_income > 0 else 0.0

    return {
        "federal": federal_tax,
        "state": state_tax,
        "credits": total_credit,
        "total": total_tax,
        "effective_rate": effective_rate,
    }


# =============================================================================
# MULTI-ASSET FORECAST
# =============================================================================
def calculate_forecast_multi_asset(
    current_age: int,
    retire_age: int,
    life_expectancy: int,
    annual_spend_today: float,
    inflation_rate: float,
    ss_start_age: int,
    social_security_annual_today: float,
    annual_contribution: float,
    pre_retire_return: float,
    post_retire_return: float,
    cash_bal: float,
    bonds_bal: float,
    etfs_bal: float,
    k401_bal: float,
    cash_yield: float,
    bonds_yield: float,
    etfs_yield: float,
    k401_yield: float,
    flow_mode: str = "pro_rata",  # "pro_rata" or "cash_first"
):
    max_age = life_expectancy
    total_months = max(0, (max_age - current_age) * 12)
    retirement_month = max(0, (retire_age - current_age) * 12)

    m_infl = inflation_rate / 12.0
    m_cash = cash_yield / 12.0
    m_bonds = bonds_yield / 12.0
    m_etfs = etfs_yield / 12.0
    m_k401 = k401_yield / 12.0

    m_spend = annual_spend_today / 12.0
    m_ss = social_security_annual_today / 12.0
    m_contrib = annual_contribution / 12.0

    cash = float(max(0, cash_bal))
    bonds = float(max(0, bonds_bal))
    etfs = float(max(0, etfs_bal))
    k401 = float(max(0, k401_bal))

    def total_pool():
        return cash + bonds + etfs + k401

    def allocate_surplus(amount: float):
        nonlocal cash, bonds, etfs, k401
        if amount <= 0:
            return
        pool = total_pool()
        if pool <= 0:
            add = amount / 4.0
            cash += add
            bonds += add
            etfs += add
            k401 += add
            return

        if flow_mode == "pro_rata":
            cash += amount * (cash / pool) if cash > 0 else 0
            bonds += amount * (bonds / pool) if bonds > 0 else 0
            etfs += amount * (etfs / pool) if etfs > 0 else 0
            k401 += amount * (k401 / pool) if k401 > 0 else 0
        else:
            cash += amount

    def withdraw_deficit(amount: float):
        nonlocal cash, bonds, etfs, k401
        if amount <= 0:
            return

        if flow_mode == "cash_first":
            for name in ["cash", "bonds", "etfs", "k401"]:
                bal = {"cash": cash, "bonds": bonds, "etfs": etfs, "k401": k401}[name]
                if amount <= 0:
                    break
                take = min(amount, bal)
                amount -= take
                if name == "cash":
                    cash -= take
                if name == "bonds":
                    bonds -= take
                if name == "etfs":
                    etfs -= take
                if name == "k401":
                    k401 -= take
        else:
            pool = total_pool()
            if pool <= 0:
                return
            w = min(amount, pool)
            ratio = w / pool
            cash -= cash * ratio
            bonds -= bonds * ratio
            etfs -= etfs * ratio
            k401 -= k401 * ratio

        cash = max(0.0, cash)
        bonds = max(0.0, bonds)
        etfs = max(0.0, etfs)
        k401 = max(0.0, k401)

    rows = []
    rows.append(
        {
            "Age": current_age,
            "Is Retired": current_age >= retire_age,
            "Required Spend": annual_spend_today,
            "Guaranteed Income": 0.0 if current_age < ss_start_age else social_security_annual_today,
            "Portfolio Withdrawal": 0.0,
            "Cash": cash,
            "Bonds": bonds,
            "ETFs": etfs,
            "401k": k401,
            "End Balance": total_pool(),
        }
    )

    for month in range(1, total_months + 1):
        sim_age = current_age + month / 12.0
        age_int = int(np.floor(sim_age))
        is_retired = month >= retirement_month

        m_spend *= (1.0 + m_infl)
        m_ss *= (1.0 + m_infl)

        cash *= (1.0 + m_cash)
        bonds *= (1.0 + m_bonds)
        etfs *= (1.0 + m_etfs)
        k401 *= (1.0 + m_k401)

        guaranteed_month = m_ss if sim_age >= ss_start_age else 0.0

        if not is_retired and m_contrib > 0:
            allocate_surplus(m_contrib)

        monthly_need = 0.0
        if is_retired:
            monthly_need = max(0.0, m_spend - guaranteed_month)
            withdraw_deficit(monthly_need)

        if month % 12 == 0:
            rows.append(
                {
                    "Age": age_int,
                    "Is Retired": age_int >= retire_age,
                    "Required Spend": m_spend * 12.0,
                    "Guaranteed Income": guaranteed_month * 12.0,
                    "Portfolio Withdrawal": monthly_need * 12.0,
                    "Cash": cash,
                    "Bonds": bonds,
                    "ETFs": etfs,
                    "401k": k401,
                    "End Balance": total_pool(),
                }
            )

        if total_pool() <= 0:
            # Record a depletion row even if depletion occurs mid-year so KPIs/Compare show the correct depletion age.
            depleted_age = age_int
            row_payload = {
                "Age": depleted_age,
                "Is Retired": depleted_age >= retire_age,
                "Required Spend": m_spend * 12.0,
                "Guaranteed Income": guaranteed_month * 12.0,
                "Portfolio Withdrawal": monthly_need * 12.0,
                "Cash": 0.0,
                "Bonds": 0.0,
                "ETFs": 0.0,
                "401k": 0.0,
                "End Balance": 0.0,
            }
            if rows and rows[-1].get("Age") == depleted_age:
                rows[-1].update(row_payload)
            else:
                rows.append(row_payload)
            break

    return pd.DataFrame(rows)


# =============================================================================
# SCENARIO MANAGER (COMPARE) - PER USER + SAFE COPIES
# =============================================================================
def _scenario_store_key() -> str:
    user = st.session_state.get("current_user") or "default"
    return f"scenarios__{user}"


def _init_scenarios():
    key = _scenario_store_key()
    if key not in st.session_state:
        st.session_state[key] = []


def _get_scenarios():
    # Always return a deep copy so editing doesn't mutate stored objects accidentally.
    return copy.deepcopy(st.session_state.get(_scenario_store_key(), []))


def _set_scenarios(scenarios):
    # Always store a deep copy for safety.
    st.session_state[_scenario_store_key()] = copy.deepcopy(scenarios)


def get_current_inputs_snapshot() -> dict:
    """
    Snapshot current widgets via session_state keys.
    """
    return {
        "current_age": int(st.session_state.get("current_age", 50)),
        "retire_age": int(st.session_state.get("retire_age", 60)),
        "life_expectancy": int(st.session_state.get("life_expectancy", 95)),
        "current_portfolio": float(st.session_state.get("current_portfolio", 1_239_000)),
        "annual_contribution": float(st.session_state.get("annual_contribution", 65_000)),
        "annual_spend_retirement": float(st.session_state.get("annual_spend_retirement", 155_000)),
        "use_multi_asset": bool(st.session_state.get("use_multi_asset", True)),
        "cash_bal": float(st.session_state.get("cash_bal", 200_000)),
        "cash_yield": float(st.session_state.get("cash_yield", 0.04)),
        "bonds_bal": float(st.session_state.get("bonds_bal", 400_000)),
        "bonds_yield": float(st.session_state.get("bonds_yield", 0.05)),
        "etfs_bal": float(st.session_state.get("etfs_bal", 439_000)),
        "etfs_yield": float(st.session_state.get("etfs_yield", 0.07)),
        "k401_bal": float(st.session_state.get("k401_bal", 200_000)),
        "k401_yield": float(st.session_state.get("k401_yield", 0.07)),
        "annual_gross_income": float(st.session_state.get("annual_gross_income", 300_000)),
        "filing_status": st.session_state.get("filing_status", "married"),
        "state_code": st.session_state.get("state_code", "NJ"),
        "manual_state_rate": float(st.session_state.get("manual_state_rate", 0.0)),
        "dependents": int(st.session_state.get("dependents", 0)),
        "annual_expenses": float(st.session_state.get("annual_expenses", 200_000)),
        "inflation_rate": float(st.session_state.get("inflation_rate", 0.03)),
        "pre_retire_return": float(st.session_state.get("pre_retire_return", 0.07)),
        "post_retire_return": float(st.session_state.get("post_retire_return", 0.045)),
        "social_security": float(st.session_state.get("social_security", 30_000)),
        "ss_start_age": int(st.session_state.get("ss_start_age", 67)),
        "flow_mode": st.session_state.get("flow_mode", "cash_first"),
    }


def run_projection_from_snapshot(s: dict) -> pd.DataFrame:
    # Normalize units on every run.
    s = normalize_snapshot(s)

    if s.get("use_multi_asset", True):
        return calculate_forecast_multi_asset(
            current_age=s["current_age"],
            retire_age=s["retire_age"],
            life_expectancy=s["life_expectancy"],
            annual_spend_today=s["annual_spend_retirement"],
            inflation_rate=s["inflation_rate"],
            ss_start_age=s["ss_start_age"],
            social_security_annual_today=s["social_security"],
            annual_contribution=s["annual_contribution"],
            pre_retire_return=s["pre_retire_return"],
            post_retire_return=s["post_retire_return"],
            cash_bal=s.get("cash_bal", 0.0),
            bonds_bal=s.get("bonds_bal", 0.0),
            etfs_bal=s.get("etfs_bal", 0.0),
            k401_bal=s.get("k401_bal", 0.0),
            cash_yield=s.get("cash_yield", 0.0),
            bonds_yield=s.get("bonds_yield", 0.0),
            etfs_yield=s.get("etfs_yield", 0.0),
            k401_yield=s.get("k401_yield", 0.0),
            flow_mode=s.get("flow_mode", "cash_first"),
        )

    # Single-portfolio model
    years = range(s["current_age"], s["life_expectancy"] + 1)
    data = []
    portfolio = float(s.get("current_portfolio", 0.0))
    running_spend_needs = float(s["annual_spend_retirement"])

    for age in years:
        is_retired = age >= s["retire_age"]

        if age > s["current_age"]:
            running_spend_needs *= (1.0 + s["inflation_rate"])

        guaranteed_income = 0.0
        if age >= s["ss_start_age"]:
            guaranteed_income = float(s["social_security"]) * ((1.0 + s["inflation_rate"]) ** (age - s["current_age"]))

        flexible_income_needed = max(0.0, running_spend_needs - guaranteed_income) if is_retired else 0.0

        start_bal = portfolio
        growth_rate = s["post_retire_return"] if is_retired else s["pre_retire_return"]
        contribution = float(s.get("annual_contribution", 0.0)) if not is_retired else 0.0

        end_bal = (start_bal + contribution - flexible_income_needed) * (1.0 + growth_rate)
        end_bal = max(0.0, end_bal)

        data.append(
            {
                "Age": age,
                "Is Retired": is_retired,
                "Portfolio Start": start_bal,
                "Required Spend": running_spend_needs,
                "Guaranteed Income": guaranteed_income,
                "Portfolio Withdrawal": flexible_income_needed,
                "End Balance": end_bal,
            }
        )

        portfolio = end_bal

    return pd.DataFrame(data)


def scenario_kpis(df: pd.DataFrame, retire_age: int, current_age: int, life_expectancy: int) -> dict:
    """Compute headline KPIs for a scenario, with robust depletion detection.

    Depletion is detected when:
      - End Balance falls to ~0 (<= $1), OR
      - the projection ends before life_expectancy (e.g., multi-asset sim stops early).
    """
    if df is None or df.empty:
        return {
            "Assets @ Retire": 0.0,
            "Final Balance": 0.0,
            "Depletion Age": "",
            "Withdrawal Rate (1st yr)": 0.0,
            "Sustainability": f"Depleted @ {current_age}",
        }

    eps = 1.0  # treat <= $1 as depleted to avoid float noise

    last_row = df.iloc[-1]
    final_balance = float(last_row.get("End Balance", 0.0))
    last_age = int(float(last_row.get("Age", life_expectancy)))

    # Assets at retirement (prefer Portfolio Start when present; else End Balance)
    retire_row = df[df["Age"] == retire_age]
    if not retire_row.empty:
        rr = retire_row.iloc[0]
        assets_at_retirement = float(rr.get("Portfolio Start", rr.get("End Balance", 0.0)))
    else:
        assets_at_retirement = 0.0

    # Depletion age: first age where End Balance ~0
    depletion_age = None
    depleted_rows = df[(df["Age"] > current_age) & (df["End Balance"] <= eps)]
    if not depleted_rows.empty:
        depletion_age = int(float(depleted_rows["Age"].min()))
    else:
        # If the series ends before the horizon, treat the final age as depletion.
        if last_age < int(life_expectancy):
            depletion_age = last_age

    # First-year withdrawal rate at retirement
    retired_rows = df[df["Age"] >= retire_age]
    if not retired_rows.empty:
        first_ret = retired_rows.iloc[0]
        withdrawal = float(first_ret.get("Portfolio Withdrawal", 0.0))
        base = float(first_ret.get("Portfolio Start", first_ret.get("End Balance", 0.0)))
        wr = withdrawal / base if base > 0 else 0.0
    else:
        wr = 0.0

    sustainability = f"Depleted @ {depletion_age}" if depletion_age is not None else f"Sustainable to {life_expectancy}"

    return {
        "Assets @ Retire": assets_at_retirement,
        "Final Balance": final_balance,
        "Depletion Age": depletion_age if depletion_age is not None else "",
        "Withdrawal Rate (1st yr)": wr,
        "Sustainability": sustainability,
    }


_init_scenarios()

# =============================================================================
# TITLE & INTRO
# =============================================================================
st.title("Strategic Retirement Planner: Cashflow & Buckets")
st.markdown(
    "Use this tool to test retirement readiness with **FIRE rules of thumb**, "
    "**cashflow projections**, and a **3-bucket investment framework**."
)
st.markdown("---")

# =============================================================================
# TABS: SINGLE vs COMPARE
# =============================================================================
tab1, tab2 = st.tabs(["Single Scenario", "Compare Scenarios"])

# =============================================================================
# TAB 1: SINGLE SCENARIO
# =============================================================================
with tab1:
    st.sidebar.header("1. Demographics & Status")
    current_age = st.sidebar.number_input("Current Age", 35, 90, 50, key="current_age")
    retire_age = st.sidebar.number_input("Retirement Age", 35, 90, 60, key="retire_age")
    life_expectancy = st.sidebar.number_input("Life Expectancy", 70, 110, 95, key="life_expectancy")

    st.sidebar.header("2. Financials (Current)")
    current_portfolio = st.sidebar.number_input("Total Invested Assets ($)", value=1_239_000, key="current_portfolio")
    annual_contribution = st.sidebar.number_input("Annual Contribution until Retirement ($)", value=65_000, key="annual_contribution")
    annual_spend_retirement = st.sidebar.number_input(
        "Desired Annual Spend in Retirement (Today's $)", value=155_000, key="annual_spend_retirement"
    )

    st.sidebar.header("2B. Portfolio Composition (Optional Multi-Asset)")
    use_multi_asset = st.sidebar.checkbox(
        "Use Multi-Asset Portfolio (Cash/Bonds/ETFs/401k)",
        value=True,
        help="When enabled, the model tracks each bucket separately and shows a stacked chart.",
        key="use_multi_asset",
    )

    with st.sidebar:
        flow_mode = st.selectbox(
            "Withdrawal Mode",
            options=["cash_first", "pro_rata"],
            index=0,
            help="cash_first withdraws from Cash→Bonds→ETFs→401k. pro_rata withdraws proportionally.",
            key="flow_mode",
        )

    if use_multi_asset:
        st.sidebar.caption("Balances should roughly sum to Total Invested Assets (above). Yields are annual %.")

        cash_bal = st.sidebar.number_input("Cash Balance ($)", value=200_000, step=10_000, key="cash_bal")
        cash_yield = st.sidebar.slider("Cash Yield (%)", 0.0, 8.0, 4.0, 0.1, key="cash_yield") / 100.0

        bonds_bal = st.sidebar.number_input("Bonds/Munis Balance ($)", value=400_000, step=10_000, key="bonds_bal")
        bonds_yield = st.sidebar.slider("Bonds Yield (%)", 0.0, 10.0, 5.0, 0.1, key="bonds_yield") / 100.0

        etfs_bal = st.sidebar.number_input("ETFs Balance ($)", value=439_000, step=10_000, key="etfs_bal")
        etfs_yield = st.sidebar.slider("ETFs Return (%)", 0.0, 12.0, 7.0, 0.1, key="etfs_yield") / 100.0

        k401_bal = st.sidebar.number_input("401k Balance ($)", value=200_000, step=10_000, key="k401_bal")
        k401_yield = st.sidebar.slider("401k Return (%)", 0.0, 12.0, 7.0, 0.1, key="k401_yield") / 100.0

        buckets_sum = cash_bal + bonds_bal + etfs_bal + k401_bal
        if abs(buckets_sum - current_portfolio) > 50_000:
            st.sidebar.warning(
                f"Bucket sum (${buckets_sum:,.0f}) differs from Total Invested Assets (${current_portfolio:,.0f}). "
                "This is OK for experimentation, but totals may look inconsistent."
            )
    else:
        cash_bal = bonds_bal = etfs_bal = k401_bal = 0.0
        cash_yield = bonds_yield = etfs_yield = k401_yield = 0.0

    st.sidebar.header("3. Tax Profile (Current Income)")
    annual_gross_income = st.sidebar.number_input("Annual Gross Income (Pre-Tax $)", value=300_000, key="annual_gross_income")

    filing_status = st.sidebar.selectbox(
        "Filing Status",
        options=["single", "married"],
        format_func=lambda x: "Single" if x == "single" else "Married Filing Jointly",
        key="filing_status",
    )

    state_code = st.sidebar.selectbox("State", options=["NJ", "Other"], index=0, key="state_code")

    manual_state_rate = 0.0
    if state_code == "Other":
        manual_state_rate = st.sidebar.slider("Other State Effective Tax Rate (%)", 0.0, 15.0, 5.0, 0.5, key="manual_state_rate")
    else:
        st.session_state["manual_state_rate"] = 0.0

    dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 0, key="dependents")

    st.sidebar.header("4. Household Expenses & Cashflow")
    annual_expenses = st.sidebar.number_input("Annual Expenses (Today's $)", value=200_000, key="annual_expenses")

    st.sidebar.header("5. Macro & Return Assumptions")
    inflation_rate = st.sidebar.slider("Inflation Rate (%)", 1.0, 5.0, 3.0, key="inflation_rate") / 100.0
    pre_retire_return = st.sidebar.slider("Pre-Retirement Growth (%)", 1.0, 12.0, 7.0, key="pre_retire_return") / 100.0
    post_retire_return = st.sidebar.slider("Post-Retirement Growth (Avg) (%)", 1.0, 10.0, 4.5, key="post_retire_return") / 100.0

    st.sidebar.header("6. Guaranteed Income (Retirement)")
    social_security = st.sidebar.number_input("Social Security/Pension (Annual $)", value=30_000, key="social_security")
    ss_start_age = st.sidebar.number_input("SS/Pension Start Age", 60, 75, 67, key="ss_start_age")

    # Tax snapshot & household surplus
    tax_info = calculate_annual_taxes(
        gross_income=annual_gross_income,
        status=filing_status,
        state_code=state_code,
        manual_state_rate=float(st.session_state.get("manual_state_rate", manual_state_rate)),
        dependents=dependents,
    )
    effective_tax_rate = tax_info["effective_rate"]
    net_take_home = annual_gross_income - tax_info["total"]
    surplus = net_take_home - annual_expenses

    st.subheader("Tax & Cashflow Snapshot")
    col_tx1, col_tx2, col_tx3, col_tx4 = st.columns(4)
    with col_tx1:
        st.metric("Gross Income", f"${annual_gross_income:,.0f}")
    with col_tx2:
        st.metric("Total Tax", f"${tax_info['total']:,.0f}")
    with col_tx3:
        st.metric("Effective Tax Rate", f"{effective_tax_rate * 100:,.1f}%")
    with col_tx4:
        st.metric("Net Take-Home Income", f"${net_take_home:,.0f}")

    st.markdown("")
    col_cash1, col_cash2 = st.columns([2, 1])
    with col_cash1:
        st.markdown("#### Tax Breakdown")
        st.markdown(
            f"- **Federal Tax (est.):** ${tax_info['federal']:,.0f}  \n"
            f"- **State Tax ({state_code}):** ${tax_info['state']:,.0f}"
        )
        if tax_info["credits"] > 0:
            st.markdown(f"- **Child Tax Credits (approx.):** ${tax_info['credits']:,.0f}")

    with col_cash2:
        st.markdown("#### Net Surplus View")
        st.metric("Annual Expenses", f"${annual_expenses:,.0f}")
        st.metric("Net Surplus (Saved)", f"${surplus:,.0f}", delta=None)

    st.caption(
        "Tax and cashflow snapshot is based on current gross income, filing status, state, dependents, "
        "and self-reported annual expenses. It is an approximation for planning, not a filing calculation."
    )
    st.markdown("---")

    # Core retirement calculations
    if use_multi_asset:
        df = calculate_forecast_multi_asset(
            current_age=current_age,
            retire_age=retire_age,
            life_expectancy=life_expectancy,
            annual_spend_today=annual_spend_retirement,
            inflation_rate=inflation_rate,
            ss_start_age=ss_start_age,
            social_security_annual_today=social_security,
            annual_contribution=annual_contribution,
            pre_retire_return=pre_retire_return,
            post_retire_return=post_retire_return,
            cash_bal=cash_bal,
            bonds_bal=bonds_bal,
            etfs_bal=etfs_bal,
            k401_bal=k401_bal,
            cash_yield=cash_yield,
            bonds_yield=bonds_yield,
            etfs_yield=etfs_yield,
            k401_yield=k401_yield,
            flow_mode=flow_mode,
        )
    else:
        years = range(current_age, life_expectancy + 1)
        data = []
        portfolio = current_portfolio
        running_spend_needs = annual_spend_retirement

        for age in years:
            is_retired = age >= retire_age
            if age > current_age:
                running_spend_needs *= (1 + inflation_rate)

            guaranteed_income = 0.0
            if age >= ss_start_age:
                guaranteed_income = social_security * ((1 + inflation_rate) ** (age - current_age))

            flexible_income_needed = max(0.0, running_spend_needs - guaranteed_income) if is_retired else 0.0

            start_bal = portfolio
            growth_rate = post_retire_return if is_retired else pre_retire_return
            contribution = annual_contribution if not is_retired else 0.0

            end_bal = (start_bal + contribution - flexible_income_needed) * (1 + growth_rate)
            end_bal = max(0.0, end_bal)

            data.append(
                {
                    "Age": age,
                    "Is Retired": is_retired,
                    "Portfolio Start": start_bal,
                    "Required Spend": running_spend_needs,
                    "Guaranteed Income": guaranteed_income,
                    "Portfolio Withdrawal": flexible_income_needed,
                    "End Balance": end_bal,
                }
            )
            portfolio = end_bal

        df = pd.DataFrame(data)

    # FIRE overview
    st.header("1. FIRE Targets (Rule of Thumb)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Standard FIRE (25x)", f"${annual_spend_retirement * 25:,.0f}", help="Target based on ~4% withdrawal rate.")
    with col2:
        st.metric("Fat/Safe FIRE (33x)", f"${annual_spend_retirement * 33:,.0f}", help="More conservative target based on ~3% withdrawal rate.")
    with col3:
        gap_25x = (annual_spend_retirement * 25) - current_portfolio
        st.metric("Gap to 25x", f"${gap_25x:,.0f}", help="Positive number indicates how much more capital is needed to reach 25x.")

    st.caption("These rules of thumb provide a quick readiness check before looking at detailed cashflow modeling.")
    st.markdown("---")

    # Cashflow & longevity model
    st.header("2. Cashflow & Longevity Model")

    retirement_row = df[df["Age"] == retire_age]
    retirement_row = retirement_row.iloc[0] if not retirement_row.empty else None

    last_row = df.iloc[-1]
    depletion_rows = df[(df["End Balance"] <= 0) & (df["Age"] > current_age)]
    depletion_age = int(depletion_rows["Age"].min()) if not depletion_rows.empty else None

    if retirement_row is not None:
        assets_at_retirement = float(retirement_row["Portfolio Start"]) if "Portfolio Start" in retirement_row else float(retirement_row["End Balance"])
        expense_at_retirement = float(retirement_row["Required Spend"])
    else:
        assets_at_retirement = 0.0
        expense_at_retirement = 0.0

    final_balance = float(last_row["End Balance"])

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(f"Total Assets @ Age {retire_age}", f"${assets_at_retirement:,.0f}")
    with m2:
        st.metric(f"Projected Annual Spend @ Age {retire_age}", f"${expense_at_retirement:,.0f}")
    with m3:
        st.metric(f"Final Balance @ Age {int(last_row['Age'])}", f"${final_balance:,.0f}")
    with m4:
        if depletion_age is not None:
            st.error(f"Sustainability: Depleted @ Age {depletion_age}")
        else:
            st.success(f"Sustainability: Sustainable to {life_expectancy}")

    fig, ax = plt.subplots(figsize=(10, 5))
    if use_multi_asset and all(col in df.columns for col in ["Cash", "Bonds", "ETFs", "401k"]):
        ax.stackplot(
            df["Age"],
            df["Cash"],
            df["Bonds"],
            df["ETFs"],
            df["401k"],
            labels=["Cash", "Bonds/Munis", "ETFs", "401k"],
            alpha=0.85,
        )
        ax.legend(loc="upper left")
    else:
        ax.plot(df["Age"], df["End Balance"], label="Portfolio Balance", linewidth=2)
        ax.legend(loc="upper right")

    ax.axvline(retire_age, linestyle="--", linewidth=1.5, label="Retirement")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Age")
    st.pyplot(fig)

    if final_balance > 0:
        st.success(f"At age {life_expectancy}, the projected portfolio balance is **${final_balance:,.0f}**.")
    else:
        st.error(f"Portfolio is projected to deplete at age **{depletion_age if depletion_age is not None else 'N/A'}** under current assumptions.")

    with st.expander("Show yearly projection table"):
        display_df = df.copy()
        money_cols = [
            "Portfolio Start",
            "Required Spend",
            "Guaranteed Income",
            "Portfolio Withdrawal",
            "End Balance",
            "Cash",
            "Bonds",
            "ETFs",
            "401k",
        ]
        for col in money_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(0).astype(int)

        st.dataframe(display_df, use_container_width=True)

    st.caption("This projection is deterministic and uses constant return and inflation assumptions. It is a planning tool, not a guarantee.")

    # 3-Bucket strategy
    st.header("3. The 3-Bucket Strategy Implementation")
    st.markdown(
        "Segment the portfolio into time-based buckets to manage **sequence-of-returns risk** "
        "and support smoother withdrawals."
    )

    if current_portfolio > 0 and retirement_row is not None:
        annual_draw_at_retire = float(retirement_row.get("Portfolio Withdrawal", 0.0))

        bucket_1_target = annual_draw_at_retire * 5
        bucket_2_target = annual_draw_at_retire * 10

        total_assets_for_buckets = float(retirement_row.get("Portfolio Start", retirement_row.get("End Balance", 0.0)))
        bucket_3_target = max(0.0, total_assets_for_buckets - bucket_1_target - bucket_2_target)

        if current_age < retire_age:
            bucket_1_target = 0.15 * current_portfolio
            bucket_2_target = 0.35 * current_portfolio
            bucket_3_target = 0.50 * current_portfolio

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            st.subheader("Bucket 1: Cash / Munis")
            st.markdown("**Role:** Years 1–5 withdrawals")
            st.info(f"Illustrative Allocation: **${bucket_1_target:,.0f}**")
            st.caption("Target: High liquidity, low volatility.")

        with col_b2:
            st.subheader("Bucket 2: Income")
            st.markdown("**Role:** Years 6–15 withdrawals")
            st.warning(f"Illustrative Allocation: **${bucket_2_target:,.0f}**")
            st.caption("Target: Stable income assets.")

        with col_b3:
            st.subheader("Bucket 3: Growth")
            st.markdown("**Role:** Year 16+ growth")
            st.error(f"Illustrative Allocation: **${bucket_3_target:,.0f}**")
            st.caption("Target: Long-term growth assets.")

        st.markdown(
            "In strong markets, **Bucket 3** gains can refill Buckets 1 and 2. "
            "In weak markets, withdrawals come from Buckets 1 and 2 to avoid forced selling."
        )
    else:
        st.warning("Portfolio value is zero or not set. Adjust inputs in the sidebar to view bucket allocations.")

    st.markdown("---")

    # Stress test
    st.header("4. Stress Test: Capacity for Loss")
    st.markdown("Simulate an immediate market shock to understand downside resilience.")

    crash_scenario = st.slider("Simulated Market Drop at Retirement (%)", 0, 50, 20, key="crash_scenario")

    if crash_scenario > 0:
        stressed_pot = current_portfolio * (1 - (crash_scenario / 100))
        st.write(f"Portfolio immediately after crash: **${stressed_pot:,.0f}**")

        if stressed_pot > (annual_spend_retirement * 25):
            st.success(
                "Even after this shock, the portfolio remains above the standard **25x FIRE** threshold. "
                "You retain a reasonable margin of safety under current assumptions."
            )
        else:
            st.warning(
                "This shock brings the portfolio **below** the 25x FIRE threshold. "
                "You may need to revisit spending, retirement age, or risk assumptions."
            )

    st.caption("This is a simple single-period stress test. In practice, you would combine this with scenario analysis and more detailed risk modeling.")

    # Plan analysis & recommendations (kept as your existing logic)
    st.markdown("---")
    st.header("5. Plan Analysis & Recommendations")

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None

    col_btn, col_help = st.columns([1, 3])
    with col_btn:
        analyze_clicked = st.button(
            "Analyze Sustainability" if st.session_state.analysis_result is None else "Refresh Analysis",
            key="analyze_button",
        )
    with col_help:
        st.caption(
            "This analysis uses your current inputs, FIRE targets, tax snapshot, cashflow, "
            "and portfolio projections to generate a high-level narrative. "
            "It is not personalized financial advice."
        )

    if analyze_clicked:
        final_balance = float(df.iloc[-1]["End Balance"])
        ends_positive = final_balance > 0

        depletion_age_2 = None
        if not ends_positive:
            zero_rows = df[df["End Balance"] == 0]
            if not zero_rows.empty:
                depletion_age_2 = int(zero_rows["Age"].min())

        retired_rows = df[df["Age"] >= retire_age]
        if not retired_rows.empty:
            first_ret_row = retired_rows.iloc[0]
            first_withdrawal = float(first_ret_row.get("Portfolio Withdrawal", 0.0))
            start_base = float(first_ret_row.get("Portfolio Start", first_ret_row.get("End Balance", 0.0)))
            initial_withdrawal_rate = (first_withdrawal / start_base) if start_base > 0 else 0.0
        else:
            first_withdrawal = 0.0
            initial_withdrawal_rate = 0.0

        if ends_positive and initial_withdrawal_rate <= 0.04:
            sustainability_label = "robust"
            sustainability_text = (
                "Based on your assumptions, the plan appears **robust**. "
                "Your portfolio is projected to last through the full planning horizon, "
                f"with an ending balance of about **${final_balance:,.0f}** and an initial withdrawal "
                f"rate of ~{initial_withdrawal_rate * 100:,.1f}%, which is in line with classical 4% guidance."
            )
        elif ends_positive and initial_withdrawal_rate <= 0.05:
            sustainability_label = "cautious"
            sustainability_text = (
                "The plan appears **generally sustainable but somewhat sensitive**. "
                "Your portfolio is projected to last through the horizon, but the initial withdrawal "
                f"rate of ~{initial_withdrawal_rate * 100:,.1f}% is above the classic 4% rule. "
                "Small changes in returns, inflation, or spending could materially impact outcomes."
            )
        else:
            sustainability_label = "at risk"
            if depletion_age_2 is not None:
                sustainability_text = (
                    "The plan appears **at risk of depletion** under current assumptions. "
                    f"Your portfolio is projected to run out around age **{depletion_age_2}**, "
                    "suggesting that retirement timing, spending levels, or risk assumptions may need revision."
                )
            else:
                sustainability_text = (
                    "The plan appears **at risk** under current assumptions. "
                    "Projected withdrawals and/or return assumptions lead to low ending balances and "
                    "a narrow margin for error."
                )

        summary_text = (
            f"You are currently **{current_age}**, planning to retire at **{retire_age}**, with an initial "
            f"retirement spending target of **${annual_spend_retirement:,.0f}** per year (in today's dollars). "
            f"Current investable assets are **${current_portfolio:,.0f}**, with assumed pre-retirement growth of "
            f"**{pre_retire_return * 100:,.1f}%**, post-retirement growth of **{post_retire_return * 100:,.1f}%**, "
            f"and inflation of **{inflation_rate * 100:,.1f}%**. "
            f"Your current tax-effective net income is about **${net_take_home:,.0f}**, with estimated annual "
            f"expenses of **${annual_expenses:,.0f}**, leaving a surplus of approximately "
            f"**${surplus:,.0f}** available for savings and flexibility."
        )

        recommendations = []
        target_25x = annual_spend_retirement * 25
        target_33x = annual_spend_retirement * 33

        if current_portfolio < target_25x:
            recommendations.append(
                f"Increase annual savings and/or redirect more of your current surplus toward investing. "
                f"Your current portfolio (~${current_portfolio:,.0f}) is below the 25x target (~${target_25x:,.0f})."
            )
        if current_portfolio < target_33x:
            recommendations.append(
                "Consider a more conservative FIRE target closer to **33x annual spending** if you want higher "
                "confidence in long-term sustainability, especially with longer life expectancy assumptions."
            )

        if surplus < 0:
            recommendations.append(
                "Your current annual expenses appear to **exceed** your after-tax income, creating a structural deficit. "
                "Addressing this gap (through spending reductions or income increases) should be a priority before "
                "relying on aggressive retirement contributions."
            )
        elif annual_contribution > surplus:
            recommendations.append(
                f"Planned annual contributions of **${annual_contribution:,.0f}** exceed the current estimated "
                f"surplus of **${surplus:,.0f}**. Validate that this contribution rate is realistic and sustainable "
                "given your lifestyle and cashflow needs."
            )

        if sustainability_label in ["cautious", "at risk"]:
            recommendations.append(
                "Evaluate retiring **later by 2–3 years** or modestly lowering initial retirement spending "
                "to improve the portfolio's ability to withstand return and inflation shocks."
            )
            recommendations.append(
                "Review your asset allocation across cash, bonds, and equities to ensure it aligns with both "
                "your risk tolerance and the need for growth to support a long retirement horizon."
            )

        if effective_tax_rate > 0.30:
            recommendations.append(
                "Explore **tax optimization strategies** (e.g., maxing tax-advantaged accounts, Roth conversions, "
                "capital gains harvesting, or efficient asset location) to improve net-of-tax returns over time."
            )

        if crash_scenario > 0:
            stressed_pot = current_portfolio * (1 - (crash_scenario / 100))
            if stressed_pot < target_25x:
                recommendations.append(
                    f"Under a {crash_scenario}% immediate market shock, investable assets fall to "
                    f"~${stressed_pot:,.0f}, below the 25x spending target. Consider holding a somewhat "
                    "larger safety bucket in cash/bonds or scaling back risk slightly pre-retirement."
                )

        if sustainability_label == "robust":
            risk_assessment = (
                "Overall portfolio risk appears **aligned** with your objectives, assuming your stated return and "
                "inflation assumptions are realistic. The main residual risks are sequence-of-returns risk in the early "
                "retirement years and potential regime shifts in inflation or tax policy."
            )
        elif sustainability_label == "cautious":
            risk_assessment = (
                "Portfolio risk appears **moderately elevated** relative to your withdrawal targets. "
                "You likely need meaningful exposure to growth assets to make the plan work, which increases sensitivity "
                "to market drawdowns, especially in the first 5–10 years of retirement."
            )
        else:
            risk_assessment = (
                "Portfolio risk and spending assumptions appear **misaligned**. At current spending levels, the plan "
                "relies on favorable markets and leaves limited margin for adverse sequences of returns or higher-than-"
                "expected inflation. De-risking without adjusting spending or timing would further compress sustainability."
            )

        st.session_state.analysis_result = {
            "summary": summary_text,
            "sustainability_check": sustainability_text,
            "recommendations": recommendations,
            "risk_assessment": risk_assessment,
        }

    result = st.session_state.analysis_result
    if result is None:
        st.info("Configure your assumptions and inputs above, then click **Analyze Sustainability** to generate a narrative assessment of your plan.")
    else:
        st.subheader("Plan Narrative")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### Executive Summary")
            st.markdown(result["summary"])
        with col_s2:
            st.markdown("#### Sustainability Check")
            st.markdown(result["sustainability_check"])

        st.markdown("")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("#### Tactical Recommendations")
            if result["recommendations"]:
                for idx, rec in enumerate(result["recommendations"], start=1):
                    st.markdown(f"**{idx}.** {rec}")
            else:
                st.markdown("No specific tactical changes are flagged by the current rule set. Monitor the plan periodically and revisit assumptions as life circumstances change.")
        with col_r2:
            st.markdown("#### Portfolio Risk Assessment")
            st.markdown(result["risk_assessment"])


# =============================================================================
# TAB 2: COMPARE SCENARIOS
# =============================================================================
with tab2:
    st.subheader("Scenario Comparison (Side-by-Side)")

    scenarios = _get_scenarios()
    if not scenarios:
        st.info("No scenarios saved yet. Create one from the Single Scenario tab by using the button below.")
    # Provide an always-available create button (even if no scenarios exist yet)
    if st.button("Create Scenario from current sidebar", use_container_width=True, key="create_scenario_top"):
        snap = normalize_snapshot(get_current_inputs_snapshot())
        scenarios = _get_scenarios()
        scenarios.append(
            {
                "id": str(uuid.uuid4())[:8],
                "name": f"Scenario {len(scenarios) + 1}",
                "inputs": copy.deepcopy(snap),
                "results_df": None,
                "kpis": None,
            }
        )
        _set_scenarios(scenarios)
        st.rerun()

    scenarios = _get_scenarios()
    if not scenarios:
        st.stop()

    # Select scenario to edit (by ID, not name)
    id_to_label = {sc["id"]: f'{sc["name"]} ({sc["id"]})' for sc in scenarios}
    ids = [sc["id"] for sc in scenarios]
    labels = [id_to_label[sid] for sid in ids]

    if "edit_scenario_id" not in st.session_state or st.session_state.edit_scenario_id not in ids:
        st.session_state.edit_scenario_id = ids[0]

    selected_label = st.selectbox(
        "Select a scenario to edit",
        options=labels,
        index=labels.index(id_to_label[st.session_state.edit_scenario_id]),
    )
    selected_id = ids[labels.index(selected_label)]
    st.session_state.edit_scenario_id = selected_id

    # Locate scenario
    sc_idx = next(i for i, sc in enumerate(scenarios) if sc["id"] == selected_id)
    scenario = scenarios[sc_idx]

    # Create / Delete row
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Duplicate selected scenario", use_container_width=True):
            scenarios = _get_scenarios()
            sc_idx = next(i for i, sc in enumerate(scenarios) if sc["id"] == selected_id)
            base = scenarios[sc_idx]
            scenarios.append(
                {
                    "id": str(uuid.uuid4())[:8],
                    "name": f"{base['name']} (copy)",
                    "inputs": copy.deepcopy(base["inputs"]),
                    "results_df": None,
                    "kpis": None,
                }
            )
            _set_scenarios(scenarios)
            st.rerun()

    with c2:
        if st.button("Delete selected scenario", use_container_width=True):
            scenarios = [sc for sc in scenarios if sc["id"] != selected_id]
            _set_scenarios(scenarios)
            st.session_state.edit_scenario_id = scenarios[0]["id"] if scenarios else None
            st.rerun()

    st.markdown("---")

    # =========================================================
    # SCENARIO EDITOR (WORKING COPY; ONLY SAVES ON SUBMIT)
    # =========================================================
    st.markdown(f"### Edit: {scenario['name']}")

    # Working deep copy (avoid mutation)
    working = normalize_snapshot(copy.deepcopy(scenario["inputs"]))
    prefix = f"sc_{selected_id}_"

    with st.form(f"edit_form_{selected_id}", clear_on_submit=False):
        new_name = st.text_input("Scenario name", value=scenario["name"], key=prefix + "name")

        st.markdown("#### Demographics")
        working["current_age"] = st.number_input("Current Age", 35, 90, int(working.get("current_age", 50)), key=prefix + "current_age")
        working["retire_age"] = st.number_input("Retirement Age", 35, 90, int(working.get("retire_age", 60)), key=prefix + "retire_age")
        working["life_expectancy"] = st.number_input("Life Expectancy", 70, 110, int(working.get("life_expectancy", 95)), key=prefix + "life_expectancy")

        st.markdown("#### Spending & Savings")
        working["annual_spend_retirement"] = st.number_input(
            "Annual spend in retirement (today $)",
            value=float(working.get("annual_spend_retirement", 155000)),
            key=prefix + "spend",
        )
        working["annual_contribution"] = st.number_input(
            "Annual contribution until retirement ($)",
            value=float(working.get("annual_contribution", 65000)),
            key=prefix + "contrib",
        )

        st.markdown("#### Assumptions (Percent)")
        infl_pct = st.slider("Inflation (%)", 1.0, 5.0, _as_percent_display(working.get("inflation_rate", 0.03), 3.0), 0.1, key=prefix + "infl_pct")
        pre_pct = st.slider("Pre-retirement return (%)", 1.0, 12.0, _as_percent_display(working.get("pre_retire_return", 0.07), 7.0), 0.1, key=prefix + "pre_pct")
        post_pct = st.slider("Post-retirement return (%)", 1.0, 10.0, _as_percent_display(working.get("post_retire_return", 0.045), 4.5), 0.1, key=prefix + "post_pct")

        working["inflation_rate"] = infl_pct / 100.0
        working["pre_retire_return"] = pre_pct / 100.0
        working["post_retire_return"] = post_pct / 100.0

        st.markdown("#### Guaranteed Income")
        working["social_security"] = st.number_input("Social Security / Pension (annual $)", value=float(working.get("social_security", 30000)), key=prefix + "ss")
        working["ss_start_age"] = st.number_input("SS / Pension start age", 60, 75, int(working.get("ss_start_age", 67)), key=prefix + "ss_age")

        st.markdown("#### Portfolio")
        working["use_multi_asset"] = st.checkbox("Use Multi-Asset (Cash/Bonds/ETFs/401k)", value=bool(working.get("use_multi_asset", True)), key=prefix + "multi")
        working["flow_mode"] = st.selectbox(
            "Withdrawal mode",
            ["cash_first", "pro_rata"],
            index=0 if working.get("flow_mode", "cash_first") == "cash_first" else 1,
            key=prefix + "flow",
        )

        if working["use_multi_asset"]:
            st.markdown("##### Multi-Asset Inputs")
            working["cash_bal"] = st.number_input("Cash balance ($)", value=float(working.get("cash_bal", 200000)), key=prefix + "cash_bal")
            cy = st.slider("Cash yield (%)", 0.0, 8.0, _as_percent_display(working.get("cash_yield", 0.04), 4.0), 0.1, key=prefix + "cash_y")
            working["cash_yield"] = cy / 100.0

            working["bonds_bal"] = st.number_input("Bonds/Munis balance ($)", value=float(working.get("bonds_bal", 400000)), key=prefix + "bonds_bal")
            by = st.slider("Bonds yield (%)", 0.0, 10.0, _as_percent_display(working.get("bonds_yield", 0.05), 5.0), 0.1, key=prefix + "bonds_y")
            working["bonds_yield"] = by / 100.0

            working["etfs_bal"] = st.number_input("ETFs balance ($)", value=float(working.get("etfs_bal", 439000)), key=prefix + "etfs_bal")
            ey = st.slider("ETFs return (%)", 0.0, 12.0, _as_percent_display(working.get("etfs_yield", 0.07), 7.0), 0.1, key=prefix + "etfs_y")
            working["etfs_yield"] = ey / 100.0

            working["k401_bal"] = st.number_input("401k balance ($)", value=float(working.get("k401_bal", 200000)), key=prefix + "k401_bal")
            ky = st.slider("401k return (%)", 0.0, 12.0, _as_percent_display(working.get("k401_yield", 0.07), 7.0), 0.1, key=prefix + "k401_y")
            working["k401_yield"] = ky / 100.0
        else:
            working["current_portfolio"] = st.number_input("Total invested assets ($)", value=float(working.get("current_portfolio", 1239000)), key=prefix + "total")

        save_clicked = st.form_submit_button("Save Scenario")

    if save_clicked:
        scenarios = _get_scenarios()
        sc_idx = next(i for i, sc in enumerate(scenarios) if sc["id"] == selected_id)

        scenarios[sc_idx]["name"] = new_name
        scenarios[sc_idx]["inputs"] = normalize_snapshot(copy.deepcopy(working))

        # Clear cached outputs so Compare tab re-runs cleanly
        scenarios[sc_idx]["results_df"] = None
        scenarios[sc_idx]["kpis"] = None

        _set_scenarios(scenarios)

        st.success("Scenario saved.")
        st.rerun()

    st.markdown("---")
    # =========================================================
    # RUN COMPARISON (recompute from saved inputs; no stale caching)
    # =========================================================
    scenarios = _get_scenarios()

    compare_ids = st.multiselect(
        "Select scenarios to compare",
        options=[sc["id"] for sc in scenarios],
        default=[sc["id"] for sc in scenarios[:3]] if len(scenarios) >= 3 else [sc["id"] for sc in scenarios[:2]],
        format_func=lambda sid: id_to_label.get(sid, sid),
    )

    # Store results in session_state so users can edit scenarios without losing the last comparison
    if "compare_results" not in st.session_state:
        st.session_state.compare_results = {}

    def _snapshot_fingerprint(snap: dict) -> str:
        # Stable hash to detect input changes
        try:
            payload = json.dumps(snap, sort_keys=True, default=str)
        except Exception:
            payload = repr(sorted(snap.items()))
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    run_clicked = st.button("Run Comparison", type="primary")

    if run_clicked:
        results = {}
        for sc in scenarios:
            if sc["id"] not in compare_ids:
                continue

            snap = normalize_snapshot(copy.deepcopy(sc["inputs"]))
            df_sc = run_projection_from_snapshot(snap)
            kpis = scenario_kpis(
                df_sc,
                retire_age=int(snap["retire_age"]),
                current_age=int(snap["current_age"]),
                life_expectancy=int(snap["life_expectancy"]),
            )

            results[sc["id"]] = {
                "name": sc["name"],
                "fingerprint": _snapshot_fingerprint(snap),
                "inputs": snap,
                "df": df_sc,
                "kpis": kpis,
            }

        st.session_state.compare_results = results
        st.success("Comparison updated.")

    # Display: always compute fresh if inputs changed since last run
    chosen = [sc for sc in scenarios if sc["id"] in compare_ids]
    if not chosen:
        st.info("Select one or more scenarios to compare.")
        st.stop()

    display_rows = []
    series = []

    for sc in chosen:
        sid = sc["id"]
        snap = normalize_snapshot(copy.deepcopy(sc["inputs"]))
        fp = _snapshot_fingerprint(snap)

        cached = st.session_state.compare_results.get(sid)
        if cached is None or cached.get("fingerprint") != fp:
            # Compute on the fly so results always match the saved scenario inputs
            df_sc = run_projection_from_snapshot(snap)
            kpis = scenario_kpis(
                df_sc,
                retire_age=int(snap["retire_age"]),
                current_age=int(snap["current_age"]),
                life_expectancy=int(snap["life_expectancy"]),
            )
            cached = {"name": sc["name"], "df": df_sc, "kpis": kpis, "fingerprint": fp}
            st.session_state.compare_results[sid] = cached

        row = {"Scenario": cached["name"]}
        row.update(cached["kpis"])
        display_rows.append(row)
        series.append((cached["name"], cached["df"]))

    kpi_df = pd.DataFrame(display_rows)

    # ---- Formatting (1 decimal for $; % for rates) ----
    money_cols = [c for c in ["Assets @ Retire", "Final Balance"] if c in kpi_df.columns]
    for c in money_cols:
        kpi_df[c] = kpi_df[c].astype(float).map(lambda x: f"${x:,.1f}")

    if "Withdrawal Rate (1st yr)" in kpi_df.columns:
        kpi_df["Withdrawal Rate (1st yr)"] = kpi_df["Withdrawal Rate (1st yr)"].astype(float).map(lambda x: f"{x*100:.2f}%")

    # Depletion Age: show blank if sustainable, else the age
    if "Depletion Age" in kpi_df.columns:
        kpi_df["Depletion Age"] = kpi_df["Depletion Age"].replace({None: "", np.nan: ""})

    st.dataframe(kpi_df, use_container_width=True, hide_index=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df_sc in series:
        ax.plot(df_sc["Age"], df_sc["End Balance"], linewidth=2, label=name)
    ax.set_xlabel("Age")
    ax.set_ylabel("Total Portfolio ($)")
    ax.legend(loc="upper right")
    st.pyplot(fig)
