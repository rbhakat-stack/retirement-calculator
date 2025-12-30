import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- APP CONFIGURATION ---
st.set_page_config(
    page_title="Strategic Retirement Planner",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- GLOBAL STYLE OVERRIDES ---
st.markdown("""
    <style>
    /* Make the central content narrower and less "blog-like" */
    .main .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Reduce heading sizes so they feel more "report-like" */
    h1 {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
    }
    h2 {
        font-size: 1.25rem !important;
        margin-top: 1.2rem !important;
        margin-bottom: 0.4rem !important;
    }
    h3 {
        font-size: 1.05rem !important;
        margin-top: 0.8rem !important;
    }

    /* Metrics: smaller numbers, tighter layout */
    [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        color: #6B7280 !important;  /* neutral grey */
    }

    /* Sidebar: slightly smaller text */
    [data-testid="stSidebar"] {
        font-size: 0.9rem !important;
    }

    /* General text */
    .stMarkdown p {
        font-size: 0.9rem;
        line-height: 1.5;
    }

    </style>
""", unsafe_allow_html=True)

# --- TITLE & INTRO ---
st.title("Strategic Retirement Planner: Cashflow & Buckets")
st.markdown(
    "Use this tool to test retirement readiness with **FIRE rules of thumb**, "
    "**cashflow projections**, and a **3-bucket investment framework**."
)

st.markdown("---")

# ============================================================
# TAX LOGIC (PYTHON VERSION OF YOUR TYPESCRIPT FUNCTIONS)
# ============================================================

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
    """
    Python version of your calculateAnnualTaxes logic.
    status: "single" or "married"
    state_code: "NJ" or "Other"
    manual_state_rate: percentage for non-NJ (e.g., 5 for 5%)
    """
    # Federal standard deduction (2024)
    standard_deduction = 14_600 if status == "single" else 29_200
    federal_taxable_income = max(0.0, gross_income - standard_deduction)

    federal_tax = calculate_progressive_tax(
        federal_taxable_income, FEDERAL_BRACKETS[status]
    )

    # Child tax credit (approximate)
    credit_phase_out_start = 400_000 if status == "married" else 200_000
    total_credit = dependents * 2_000

    if gross_income > credit_phase_out_start:
        reduction = np.ceil((gross_income - credit_phase_out_start) / 1_000) * 50
        total_credit = max(0.0, total_credit - reduction)

    federal_tax = max(0.0, federal_tax - total_credit)

    # State tax
    if state_code == "NJ":
        # Minimal NJ exemptions
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


# ============================================================
# SIDEBAR INPUTS
# ============================================================

st.sidebar.header("1. Demographics & Status")
current_age = st.sidebar.number_input("Current Age", 35, 90, 50)
retire_age = st.sidebar.number_input("Retirement Age", 35, 90, 60)
life_expectancy = st.sidebar.number_input("Life Expectancy", 70, 110, 95)

st.sidebar.header("2. Financials (Current)")
current_portfolio = st.sidebar.number_input("Total Invested Assets ($)", value=1_239_000)
annual_contribution = st.sidebar.number_input(
    "Annual Contribution until Retirement ($)", value=65_000
)
annual_spend_retirement = st.sidebar.number_input(
    "Desired Annual Spend in Retirement (Today's $)", value=155_000
)


st.sidebar.header("2B. Portfolio Composition (Optional Multi-Asset)")

use_multi_asset = st.sidebar.checkbox(
    "Use Multi-Asset Portfolio (Cash/Bonds/ETFs/401k)", value=True,
    help="When enabled, the model tracks each bucket separately and shows a stacked chart."
)

if use_multi_asset:
    st.sidebar.caption("Balances should roughly sum to Total Invested Assets (above). Yields are annual %.")

    cash_bal = st.sidebar.number_input("Cash Balance ($)", value=200_000, step=10_000)
    cash_yield = st.sidebar.slider("Cash Yield (%)", 0.0, 8.0, 4.0, 0.1) / 100

    bonds_bal = st.sidebar.number_input("Bonds/Munis Balance ($)", value=400_000, step=10_000)
    bonds_yield = st.sidebar.slider("Bonds Yield (%)", 0.0, 10.0, 5.0, 0.1) / 100

    etfs_bal = st.sidebar.number_input("ETFs Balance ($)", value=439_000, step=10_000)
    etfs_yield = st.sidebar.slider("ETFs Return (%)", 0.0, 12.0, 7.0, 0.1) / 100

    k401_bal = st.sidebar.number_input("401k Balance ($)", value=200_000, step=10_000)
    k401_yield = st.sidebar.slider("401k Return (%)", 0.0, 12.0, 7.0, 0.1) / 100

    # Simple consistency message
    buckets_sum = cash_bal + bonds_bal + etfs_bal + k401_bal
    if abs(buckets_sum - current_portfolio) > 50_000:
        st.sidebar.warning(
            f"Bucket sum (${buckets_sum:,.0f}) differs from Total Invested Assets (${current_portfolio:,.0f}). "
            "This is OK for experimentation, but totals may look inconsistent."
        )


st.sidebar.header("3. Tax Profile (Current Income)")
annual_gross_income = st.sidebar.number_input(
    "Annual Gross Income (Pre-Tax $)", value=300_000
)

filing_status = st.sidebar.selectbox(
    "Filing Status",
    options=["single", "married"],
    format_func=lambda x: "Single" if x == "single" else "Married Filing Jointly",
)

state_code = st.sidebar.selectbox(
    "State",
    options=["NJ", "Other"],
    index=0,
)

manual_state_rate = 0.0
if state_code == "Other":
    manual_state_rate = st.sidebar.slider(
        "Other State Effective Tax Rate (%)", 0.0, 15.0, 5.0, 0.5
    )

dependents = st.sidebar.number_input("Number of Dependents", 0, 10, 0)

st.sidebar.header("4. Household Expenses & Cashflow")
annual_expenses = st.sidebar.number_input(
    "Annual Expenses (Today's $)", value=200_000
)

st.sidebar.header("5. Macro & Return Assumptions")
inflation_rate = st.sidebar.slider("Inflation Rate (%)", 1.0, 5.0, 3.0) / 100
pre_retire_return = st.sidebar.slider("Pre-Retirement Growth (%)", 1.0, 12.0, 7.0) / 100
post_retire_return = st.sidebar.slider(
    "Post-Retirement Growth (Avg) (%)", 1.0, 10.0, 4.5
) / 100

st.sidebar.header("6. Guaranteed Income (Retirement)")
social_security = st.sidebar.number_input(
    "Social Security/Pension (Annual $)", value=30_000
)
ss_start_age = st.sidebar.number_input("SS/Pension Start Age", 60, 75, 67)

# ============================================================
# TAX SNAPSHOT & HOUSEHOLD SURPLUS
# ============================================================

tax_info = calculate_annual_taxes(
    gross_income=annual_gross_income,
    status=filing_status,
    state_code=state_code,
    manual_state_rate=manual_state_rate,
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
    st.metric(
        "Effective Tax Rate",
        f"{effective_tax_rate * 100:,.1f}%",
    )
with col_tx4:
    st.metric("Net Take-Home Income", f"${net_take_home:,.0f}")

# Detailed breakdown card-like layout
st.markdown("")
col_cash1, col_cash2 = st.columns([2, 1])

with col_cash1:
    st.markdown("#### Tax Breakdown")
    st.markdown(
        f"- **Federal Tax (est.):** ${tax_info['federal']:,.0f}  \n"
        f"- **State Tax ({state_code}):** ${tax_info['state']:,.0f}"
    )
    if tax_info["credits"] > 0:
        st.markdown(
            f"- **Child Tax Credits (approx.):** "
            f"${tax_info['credits']:,.0f}"
        )

with col_cash2:
    st.markdown("#### Net Surplus View")
    st.metric("Annual Expenses", f"${annual_expenses:,.0f}")
    st.metric(
        "Net Surplus (Saved)",
        f"${surplus:,.0f}",
        delta=None,
    )

st.caption(
    "Tax and cashflow snapshot is based on current gross income, filing status, state, dependents, "
    "and self-reported annual expenses. It is an approximation for planning, not a filing calculation."
)

st.markdown("---")

def calculate_forecast_multi_asset(
    current_age: int,
    retire_age: int,
    life_expectancy: int,
    annual_spend_today: float,
    inflation_rate: float,
    ss_start_age: int,
    social_security_annual_today: float,
    annual_contribution: float,
    # Working vs retired return assumptions (used ONLY when single-portfolio mode)
    pre_retire_return: float,
    post_retire_return: float,
    # Multi-asset buckets
    cash_bal: float,
    bonds_bal: float,
    etfs_bal: float,
    k401_bal: float,
    cash_yield: float,
    bonds_yield: float,
    etfs_yield: float,
    k401_yield: float,
    # Withdrawal / contribution allocation
    flow_mode: str = "pro_rata",  # "pro_rata" or "cash_first"
):
    """
    Monthly simulation (like your TS calculateForecast):
    - Expenses inflate monthly
    - SS inflates monthly once started
    - Growth applied monthly per bucket
    - While working: contribution added monthly (allocated based on flow_mode)
    - In retirement: withdrawals funded from buckets (flow_mode controls order)
    Returns a YEARLY dataframe aligned to your existing app sections.
    """

    # Build month horizon
    max_age = life_expectancy
    total_months = max(0, (max_age - current_age) * 12)
    retirement_month = max(0, (retire_age - current_age) * 12)

    # Monthly rates
    m_infl = inflation_rate / 12.0
    m_cash = cash_yield / 12.0
    m_bonds = bonds_yield / 12.0
    m_etfs = etfs_yield / 12.0
    m_k401 = k401_yield / 12.0

    # Monthly spending + SS (start in "today dollars")
    m_spend = annual_spend_today / 12.0
    m_ss = social_security_annual_today / 12.0

    # Monthly contribution (while working)
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
            # split evenly if empty
            add = amount / 4.0
            cash += add; bonds += add; etfs += add; k401 += add
            return
        # pro-rata add
        if flow_mode == "pro_rata":
            cash += amount * (cash / pool) if cash > 0 else 0
            bonds += amount * (bonds / pool) if bonds > 0 else 0
            etfs += amount * (etfs / pool) if etfs > 0 else 0
            k401 += amount * (k401 / pool) if k401 > 0 else 0
        else:
            # "cash_first" for deposits: still pro-rata is more realistic; keep simple
            cash += amount

    def withdraw_deficit(amount: float):
        nonlocal cash, bonds, etfs, k401
        if amount <= 0:
            return
        # Withdraw based on chosen method
        if flow_mode == "cash_first":
            # Withdraw from cash, then bonds, then etfs, then 401k
            for name in ["cash", "bonds", "etfs", "k401"]:
                bal = {"cash": cash, "bonds": bonds, "etfs": etfs, "k401": k401}[name]
                if amount <= 0:
                    break
                take = min(amount, bal)
                amount -= take
                if name == "cash": cash -= take
                if name == "bonds": bonds -= take
                if name == "etfs": etfs -= take
                if name == "k401": k401 -= take
        else:
            # pro-rata withdraw
            pool = total_pool()
            if pool <= 0:
                return
            w = min(amount, pool)
            ratio = w / pool
            cash -= cash * ratio
            bonds -= bonds * ratio
            etfs -= etfs * ratio
            k401 -= k401 * ratio

        # floor
        cash = max(0.0, cash); bonds = max(0.0, bonds); etfs = max(0.0, etfs); k401 = max(0.0, k401)

    # Yearly output aligned with your existing DF columns + new bucket columns
    rows = []
    rows.append({
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
    })

    for month in range(1, total_months + 1):
        sim_age = current_age + month / 12.0
        age_int = int(np.floor(sim_age))
        is_retired = month >= retirement_month

        # Inflate spending + SS
        m_spend *= (1.0 + m_infl)
        m_ss *= (1.0 + m_infl)

        # Apply monthly growth
        cash *= (1.0 + m_cash)
        bonds *= (1.0 + m_bonds)
        etfs *= (1.0 + m_etfs)
        k401 *= (1.0 + m_k401)

        # Determine guaranteed income (monthly)
        guaranteed_month = m_ss if sim_age >= ss_start_age else 0.0

        # Contributions pre-retirement
        if not is_retired and m_contrib > 0:
            allocate_surplus(m_contrib)

        # Retirement withdrawals (monthly)
        monthly_need = 0.0
        if is_retired:
            monthly_need = max(0.0, m_spend - guaranteed_month)
            withdraw_deficit(monthly_need)

        # Record annually
        if month % 12 == 0:
            required_annual = m_spend * 12.0
            guaranteed_annual = guaranteed_month * 12.0
            rows.append({
                "Age": age_int,
                "Is Retired": age_int >= retire_age,
                "Required Spend": required_annual,
                "Guaranteed Income": guaranteed_annual,
                "Portfolio Withdrawal": monthly_need * 12.0,  # annualized approximation at year-end
                "Cash": cash,
                "Bonds": bonds,
                "ETFs": etfs,
                "401k": k401,
                "End Balance": total_pool(),
            })

        if total_pool() <= 0:
            # stop early; still fill yearly record? keep simple: break
            break

    return pd.DataFrame(rows)


# ============================================================
# CORE RETIREMENT CALCULATIONS
# ============================================================

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
        flow_mode="cash_first",  # change to "pro_rata" if you want
    )
else:
    # --- your existing single-portfolio deterministic model ---
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

        flexible_income_needed = max(0, running_spend_needs - guaranteed_income) if is_retired else 0

        start_bal = portfolio
        growth_rate = post_retire_return if is_retired else pre_retire_return
        contribution = annual_contribution if not is_retired else 0

        end_bal = (start_bal + contribution - flexible_income_needed) * (1 + growth_rate)
        if end_bal < 0:
            end_bal = 0

        data.append({
            "Age": age,
            "Is Retired": is_retired,
            "Portfolio Start": start_bal,
            "Required Spend": running_spend_needs,
            "Guaranteed Income": guaranteed_income,
            "Portfolio Withdrawal": flexible_income_needed,
            "End Balance": end_bal,
        })

        portfolio = end_bal

    df = pd.DataFrame(data)


# ============================================================
# SECTION 1: FIRE OVERVIEW
# ============================================================

st.header("1. FIRE Targets (Rule of Thumb)")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Standard FIRE (25x)",
        f"${annual_spend_retirement * 25:,.0f}",
        help="Target based on ~4% withdrawal rate."
    )

with col2:
    st.metric(
        "Fat/Safe FIRE (33x)",
        f"${annual_spend_retirement * 33:,.0f}",
        help="More conservative target based on ~3% withdrawal rate."
    )

with col3:
    gap_25x = (annual_spend_retirement * 25) - current_portfolio
    st.metric(
        "Gap to 25x",
        f"${gap_25x:,.0f}",
        help="Positive number indicates how much more capital is needed to reach 25x."
    )

st.caption(
    "These rules of thumb provide a quick readiness check before looking at detailed cashflow modeling."
)

st.markdown("---")

# ============================================================
# SECTION 2: CASHFLOW & LONGEVITY MODEL (ENHANCED RESULTS VIEW)
# ============================================================

st.header("2. Cashflow & Longevity Model")

# --- Key points for summary metrics (like ResultsChart) ---
retirement_row = df[df["Age"] == retire_age]
retirement_row = retirement_row.iloc[0] if not retirement_row.empty else None

last_row = df.iloc[-1]
depletion_rows = df[(df["End Balance"] <= 0) & (df["Age"] > current_age)]
depletion_age = int(depletion_rows["Age"].min()) if not depletion_rows.empty else None

if retirement_row is not None:
    if "Portfolio Start" in retirement_row:
        assets_at_retirement = float(retirement_row["Portfolio Start"])
    else:
        assets_at_retirement = float(retirement_row["End Balance"])
else:
    assets_at_retirement = 0.0

expense_at_retirement = float(retirement_row["Required Spend"]) if retirement_row is not None else 0.0
final_balance = float(last_row["End Balance"])

# --- Summary metric "cards" ---
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        f"Total Assets @ Age {retire_age}",
        f"${assets_at_retirement:,.0f}",
        help="Portfolio starting balance at retirement age."
    )

with m2:
    st.metric(
        f"Projected Annual Spend @ Age {retire_age}",
        f"${expense_at_retirement:,.0f}",
        help="Inflation-adjusted annual spending requirement at retirement age."
    )

with m3:
    st.metric(
        f"Final Balance @ Age {int(last_row['Age'])}",
        f"${final_balance:,.0f}",
        help="Projected ending portfolio balance at the end of the horizon."
    )

with m4:
    if depletion_age is not None:
        st.error(f"Sustainability: Depleted @ Age {depletion_age}")
    else:
        st.success(f"Sustainability: Sustainable to {life_expectancy}")

st.markdown("")

# --- Plot (single line or stacked area depending on mode) ---
fig, ax = plt.subplots(figsize=(10, 5))

if use_multi_asset and all(col in df.columns for col in ["Cash", "Bonds", "ETFs", "401k"]):
    ax.stackplot(
        df["Age"],
        df["Cash"],
        df["Bonds"],
        df["ETFs"],
        df["401k"],
        labels=["Cash", "Bonds/Munis", "ETFs", "401k"],
        alpha=0.85
    )
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Age")
    ax.axvline(retire_age, linestyle="--", linewidth=1.5, label="Retirement")
    ax.legend(loc="upper left")
else:
    ax.plot(df["Age"], df["End Balance"], label="Portfolio Balance", linewidth=2)
    ax.axvline(retire_age, linestyle="--", linewidth=1.5, label="Retirement")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Age")
    ax.legend(loc="upper right")

st.pyplot(fig)


# Narrative alert (existing behavior, but aligned to new summary)
if final_balance > 0:
    st.success(
        f"At age {life_expectancy}, the projected portfolio balance is "
        f"**${final_balance:,.0f}**."
    )
else:
    st.error(
        f"Portfolio is projected to deplete at age **{depletion_age if depletion_age is not None else 'N/A'}** "
        f"under current assumptions."
    )

with st.expander("Show yearly projection table"):
    display_df = df.copy()

    # Format only columns that exist (prevents KeyError)
    money_cols = ["Portfolio Start", "Required Spend", "Guaranteed Income", "Portfolio Withdrawal", "End Balance",
                  "Cash", "Bonds", "ETFs", "401k"]

    for col in money_cols:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(0).astype(int)

    st.dataframe(display_df, use_container_width=True)


st.caption(
    "This projection is deterministic and uses constant return and inflation assumptions. "
    "It is a planning tool, not a guarantee."
)


# ============================================================
# SECTION 3: 3-BUCKET STRATEGY
# ============================================================

st.header("3. The 3-Bucket Strategy Implementation")
st.markdown(
    "Segment the portfolio into time-based buckets to manage **sequence-of-returns risk** "
    "and support smoother withdrawals."
)

if current_portfolio > 0:
    # Values at retirement
    retire_year_row = df[df["Age"] == retire_age].iloc[0]
    annual_draw_at_retire = retire_year_row["Portfolio Withdrawal"]

    # Bucket logic
    bucket_1_target = annual_draw_at_retire * 5      # Years 1–5
    bucket_2_target = annual_draw_at_retire * 10     # Years 6–15

    # --- FIX: use Portfolio Start if present, else End Balance ---
    if "Portfolio Start" in df.columns:
        total_assets_for_buckets = df.iloc[0]["Portfolio Start"]
    else:
        total_assets_for_buckets = df.iloc[0]["End Balance"]

    bucket_3_target = max(
        0,
        total_assets_for_buckets - bucket_1_target - bucket_2_target
    )

    # If pre-retirement, show a simplified current allocation view
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
    st.warning(
        "Portfolio value is zero or not set. Adjust inputs in the sidebar to view bucket allocations."
    )


st.markdown("---")

# ============================================================
# SECTION 4: STRESS TEST
# ============================================================

st.header("4. Stress Test: Capacity for Loss")
st.markdown("Simulate an immediate market shock to understand downside resilience.")

crash_scenario = st.slider("Simulated Market Drop at Retirement (%)", 0, 50, 20)

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

st.caption(
    "This is a simple single-period stress test. In practice, you would combine this with "
    "scenario analysis and more detailed risk modeling."
)

# ============================================================
# SECTION 5: PLAN ANALYSIS & RECOMMENDATIONS
# ============================================================

st.markdown("---")
st.header("5. Plan Analysis & Recommendations")

# Initialize analysis state
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Button row
col_btn, col_help = st.columns([1, 3])
with col_btn:
    analyze_clicked = st.button(
        "Analyze Sustainability" if st.session_state.analysis_result is None else "Refresh Analysis"
    )

with col_help:
    st.caption(
        "This analysis uses your current inputs, FIRE targets, tax snapshot, cashflow, "
        "and portfolio projections to generate a high-level narrative. "
        "It is not personalized financial advice."
    )

if analyze_clicked:
    # --- Derive key metrics from existing data ---
    final_balance = float(df.iloc[-1]["End Balance"])
    ends_positive = final_balance > 0

    # Depletion age if any
    depletion_age = None
    if not ends_positive:
        zero_rows = df[df["End Balance"] == 0]
        if not zero_rows.empty:
            depletion_age = int(zero_rows["Age"].min())

        # Effective withdrawal rate at retirement (first retired year)
    retired_rows = df[df["Age"] >= retire_age]
    if not retired_rows.empty:
        first_ret_row = retired_rows.iloc[0]
        first_withdrawal = float(first_ret_row.get("Portfolio Withdrawal", 0.0))

        # Use Portfolio Start if present (single-asset), else End Balance (multi-asset total)
        if "Portfolio Start" in df.columns:
            start_base = float(first_ret_row.get("Portfolio Start", 0.0))
        else:
            start_base = float(first_ret_row.get("End Balance", 0.0))

        initial_withdrawal_rate = (first_withdrawal / start_base) if start_base > 0 else 0.0
    else:
        first_withdrawal = 0.0
        initial_withdrawal_rate = 0.0


    # Simple classification of sustainability
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
        if depletion_age is not None:
            sustainability_text = (
                "The plan appears **at risk of depletion** under current assumptions. "
                f"Your portfolio is projected to run out around age **{depletion_age}**, "
                "suggesting that retirement timing, spending levels, or risk assumptions may need revision."
            )
        else:
            sustainability_text = (
                "The plan appears **at risk** under current assumptions. "
                "Projected withdrawals and/or return assumptions lead to low ending balances and "
                "a narrow margin for error."
            )

    # Executive summary
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

    # Recommendations (rule-based)
    recommendations = []

    # Gap to FIRE 25x / 33x
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

    # Surplus vs contributions
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

    # If plan at risk or cautious – levers
    if sustainability_label in ["cautious", "at risk"]:
        recommendations.append(
            "Evaluate retiring **later by 2–3 years** or modestly lowering initial retirement spending "
            "to improve the portfolio's ability to withstand return and inflation shocks."
        )
        recommendations.append(
            "Review your asset allocation across cash, bonds, and equities to ensure it aligns with both "
            "your risk tolerance and the need for growth to support a long retirement horizon."
        )

    # Tax optimization angle
    if effective_tax_rate > 0.30:
        recommendations.append(
            "Explore **tax optimization strategies** (e.g., maxing tax-advantaged accounts, Roth conversions, "
            "capital gains harvesting, or efficient asset location) to improve net-of-tax returns over time."
        )

    # If the stress test scenario is severe vs FIRE target
    if "stressed_pot" in locals() and crash_scenario > 0:
        if stressed_pot < target_25x:
            recommendations.append(
                f"Under a {crash_scenario}% immediate market shock, investable assets fall to "
                f"~${stressed_pot:,.0f}, below the 25x spending target. Consider holding a somewhat "
                "larger safety bucket in cash/bonds or scaling back risk slightly pre-retirement."
            )

    # Risk assessment narrative
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

    # Store in session_state
    st.session_state.analysis_result = {
        "summary": summary_text,
        "sustainability_check": sustainability_text,
        "recommendations": recommendations,
        "risk_assessment": risk_assessment,
    }

# --- Display analysis if available ---
result = st.session_state.analysis_result
if result is None:
    st.info(
        "Configure your assumptions and inputs above, then click **Analyze Sustainability** "
        "to generate a narrative assessment of your plan."
    )
else:
    st.subheader("Plan Narrative")

    # Top row: summary + sustainability
    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("#### Executive Summary")
        st.markdown(result["summary"])

    with col_s2:
        st.markdown("#### Sustainability Check")
        st.markdown(result["sustainability_check"])

    st.markdown("")

    # Second row: recommendations + risk
    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.markdown("#### Tactical Recommendations")
        if result["recommendations"]:
            for idx, rec in enumerate(result["recommendations"], start=1):
                st.markdown(f"**{idx}.** {rec}")
        else:
            st.markdown(
                "No specific tactical changes are flagged by the current rule set. "
                "Monitor the plan periodically and revisit assumptions as life circumstances change."
            )

    with col_r2:
        st.markdown("#### Portfolio Risk Assessment")
        st.markdown(result["risk_assessment"])
