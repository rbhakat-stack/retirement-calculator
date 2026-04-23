"""
Pure tax calculation logic for the Strategic Retirement Planner.

No Streamlit dependency — safe to import in tests and other contexts.
All functions are deterministic and side-effect free.

Tax brackets reflect 2024 IRS/NJ values. Update annually before January 1.
"""

import math

TAX_YEAR = 2024

FEDERAL_BRACKETS = {
    "single": [
        {"limit": 11_600,  "rate": 0.10},
        {"limit": 47_150,  "rate": 0.12},
        {"limit": 100_525, "rate": 0.22},
        {"limit": 191_950, "rate": 0.24},
        {"limit": 243_725, "rate": 0.32},
        {"limit": 609_350, "rate": 0.35},
        {"limit": float("inf"), "rate": 0.37},
    ],
    "married": [
        {"limit": 23_200,  "rate": 0.10},
        {"limit": 94_300,  "rate": 0.12},
        {"limit": 201_050, "rate": 0.22},
        {"limit": 383_900, "rate": 0.24},
        {"limit": 487_450, "rate": 0.32},
        {"limit": 731_200, "rate": 0.35},
        {"limit": float("inf"), "rate": 0.37},
    ],
}

NJ_BRACKETS = {
    "single": [
        {"limit": 20_000,    "rate": 0.014},
        {"limit": 35_000,    "rate": 0.0175},
        {"limit": 40_000,    "rate": 0.035},
        {"limit": 75_000,    "rate": 0.05525},
        {"limit": 500_000,   "rate": 0.0637},
        {"limit": 1_000_000, "rate": 0.0897},
        {"limit": float("inf"), "rate": 0.1075},
    ],
    "married": [
        {"limit": 20_000,    "rate": 0.014},
        {"limit": 50_000,    "rate": 0.0175},
        {"limit": 70_000,    "rate": 0.0245},
        {"limit": 80_000,    "rate": 0.035},
        {"limit": 150_000,   "rate": 0.05525},
        {"limit": 500_000,   "rate": 0.0637},
        {"limit": 1_000_000, "rate": 0.0897},
        {"limit": float("inf"), "rate": 0.1075},
    ],
}


def calculate_progressive_tax(taxable_income: float, brackets) -> float:
    tax = 0.0
    previous_limit = 0.0
    for bracket in brackets:
        limit = bracket["limit"]
        rate  = bracket["rate"]
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
    status: "single" or "married"
    state_code: "NJ" or "Other"
    manual_state_rate: percentage for non-NJ (e.g., 5 for 5%)
    """
    standard_deduction = 14_600 if status == "single" else 29_200
    federal_taxable_income = max(0.0, gross_income - standard_deduction)

    federal_tax = calculate_progressive_tax(
        federal_taxable_income, FEDERAL_BRACKETS[status]
    )

    credit_phase_out_start = 400_000 if status == "married" else 200_000
    total_credit = dependents * 2_000

    if gross_income > credit_phase_out_start:
        reduction = math.ceil((gross_income - credit_phase_out_start) / 1_000) * 50
        total_credit = max(0.0, total_credit - reduction)

    federal_tax = max(0.0, federal_tax - total_credit)

    if state_code == "NJ":
        nj_exempt = (dependents * 1_500) + (2_000 if status == "married" else 1_000)
        nj_taxable = max(0.0, gross_income - nj_exempt)
        state_tax = calculate_progressive_tax(nj_taxable, NJ_BRACKETS[status])
    else:
        state_tax = gross_income * (manual_state_rate / 100.0)

    total_tax = federal_tax + state_tax
    effective_rate = total_tax / gross_income if gross_income > 0 else 0.0

    return {
        "federal":        federal_tax,
        "state":          state_tax,
        "credits":        total_credit,
        "total":          total_tax,
        "effective_rate": effective_rate,
    }
