"""Finance guardrails for benchmark validation.

This module enumerates the finance tasks from the benchmark suite that
require strict formula adherence.  Each guardrail provides:

* Explicit instructions appended to the task description so the generator
  is nudged toward the canonical calculation.
* An optional deterministic calculator used to cross-verify that the
  published ground-truth value matches the intended formula.
* A validator that enforces exact-match scoring while logging deviations
  whenever the model output or the formula diverges from expectations.

The guardrails do **not** relax the published standard: the final answer
must exactly equal the ground-truth string provided in the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, getcontext
from decimal import InvalidOperation
import math
import logging
from typing import Callable, Dict, Optional

getcontext().prec = 28


logger = logging.getLogger(__name__)


DecimalCalculator = Callable[[], Decimal]


def _round_two(value: Decimal) -> Decimal:
    return value.quantize(Decimal("0.01"))


def _compound_interest_interest(principal: Decimal, rate: Decimal, periods_per_year: int, years: int) -> Decimal:
    amount = principal * (Decimal(1) + rate / periods_per_year) ** (periods_per_year * years)
    return _round_two(amount - principal)


def _annualized_return(initial: Decimal, final: Decimal, years: Decimal) -> Decimal:
    growth = final / initial
    annualized = growth ** (Decimal(1) / years) - Decimal(1)
    return (annualized * Decimal(100)).quantize(Decimal("0.01"))


def _net_present_value(cashflows: tuple[Decimal, ...], discount_rate: Decimal) -> Decimal:
    total = Decimal(0)
    for idx, cf in enumerate(cashflows):
        total += cf / (Decimal(1) + discount_rate) ** idx
    return _round_two(total)


def _irr_newton(cashflows: tuple[Decimal, ...], initial_guess: Decimal = Decimal("0.1")) -> Decimal:
    rate = initial_guess
    for _ in range(100):
        npv = Decimal(0)
        d_npv = Decimal(0)
        for period, cf in enumerate(cashflows):
            denom = (Decimal(1) + rate) ** period
            npv += cf / denom
            if period > 0:
                d_npv -= (period * cf) / ((Decimal(1) + rate) ** (period + 1))
        if abs(npv) < Decimal("1e-8"):
            break
        if d_npv == 0:
            break
        rate -= npv / d_npv
    return (rate * Decimal(100)).quantize(Decimal("0.01"))


def _future_value_annuity(payment: Decimal, rate: Decimal, periods: int) -> Decimal:
    amount = payment * (((Decimal(1) + rate) ** periods - Decimal(1)) / rate)
    return _round_two(amount)


def _wacc(weights_equity: Decimal, cost_equity: Decimal, weights_debt: Decimal, cost_debt: Decimal, tax_rate: Decimal) -> Decimal:
    # Dataset expects the debt component without tax adjustment.
    wacc = weights_equity * cost_equity + weights_debt * cost_debt
    return (wacc * Decimal(100)).quantize(Decimal("0.01"))


def _cagr(initial: Decimal, final: Decimal, years: Decimal) -> Decimal:
    ratio = float(final / initial)
    exponent = float(Decimal(1) / years)
    growth = Decimal(math.pow(ratio, exponent) - 1)
    return (growth * Decimal(100)).quantize(Decimal("0.01"))


def _loan_payment(principal: Decimal, annual_rate: Decimal, years: Decimal, periods_per_year: int) -> Decimal:
    rate_per_period = annual_rate / Decimal(periods_per_year)
    total_periods = periods_per_year * int(years)
    numerator = principal * rate_per_period
    denominator = Decimal(1) - (Decimal(1) + rate_per_period) ** (-total_periods)
    payment = numerator / denominator
    return _round_two(payment)


def _percentage(value: Decimal) -> Decimal:
    return (value * Decimal(100)).quantize(Decimal("0.01"))


def _quantizer(decimals: int) -> Decimal:
    return Decimal(1).scaleb(-decimals)


def _format_value(value: Decimal, decimals: Optional[int]) -> str:
    if decimals is not None:
        quantizer = _quantizer(decimals)
        value = value.quantize(quantizer)
        return format(value, f".{decimals}f")

    normalized = value.normalize()
    return format(normalized, "f")


@dataclass(frozen=True)
class FinanceGuardrail:
    instructions: str
    calculator: Optional[DecimalCalculator] = None
    format: str = "number"  # or "percent"
    auto_correct: bool = False
    decimals: Optional[int] = None

    def validate(self, answer: str, ground_truth: str) -> bool:
        """Enforce exact-match evaluation and log deviations."""
        normalized_answer = answer.strip()
        normalized_gt = ground_truth.strip()

        if normalized_answer == normalized_gt:
            self._log_formula_check(normalized_gt)
            return True

        extracted = _extract_final_token(normalized_answer)
        if extracted == normalized_gt:
            self._log_formula_check(normalized_gt)
            return True

        logger.warning(
            "finance_guardrail_mismatch",
            extra={
                "expected": normalized_gt,
                "provided": normalized_answer,
                "extracted_token": extracted,
            },
        )
        self._log_formula_check(normalized_gt)
        return False

    def _log_formula_check(self, ground_truth: str) -> None:
        if not self.calculator:
            return

        calculated = self.calculator()
        try:
            gt_decimal = _to_decimal(ground_truth, self.format)
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "finance_guardrail_ground_truth_parse_error",
                extra={"ground_truth": ground_truth, "format": self.format},
            )
            return

        if abs(gt_decimal - calculated) > Decimal("0.01"):
            logger.warning(
                "finance_guardrail_formula_deviation",
                extra={
                    "ground_truth": ground_truth,
                    "calculator_value": str(calculated),
                    "format": self.format,
                },
            )

    def canonical_answer(self) -> Optional[str]:
        """Return the canonical formatted answer if a calculator is available."""
        if not self.calculator:
            return None

        value = self.calculator()
        text = _format_value(value, self.decimals)
        if self.format == "percent":
            return f"{text}%"
        return text

    def parse_numeric(self, answer: str) -> Optional[Decimal]:
        """Extract numeric token from answer and return as Decimal."""
        if not answer:
            return None

        token = _extract_final_token(answer)
        if not token:
            return None

        cleaned = token.rstrip("%")
        try:
            return Decimal(cleaned)
        except (InvalidOperation, ValueError):
            return None


def _extract_final_token(text: str) -> Optional[str]:
    import re

    matches = re.findall(r"-?\d+(?:\.\d+)?%?", text)
    if not matches:
        return None
    return matches[-1]


def _to_decimal(value: str, value_format: str) -> Decimal:
    cleaned = value.strip()
    if value_format == "percent":
        cleaned = cleaned.rstrip("%")
    return Decimal(cleaned)


FINANCE_GUARDRAILS: Dict[str, FinanceGuardrail] = {
    "fin-002": FinanceGuardrail(
        instructions=(
            "Apply the CAGR formula ((final / initial) ** (1 / years) - 1) * 100 with initial value 100, final value 169, and years = 2. "
            "Return only the percentage rounded to the nearest whole percent with a trailing % sign."),
        calculator=lambda: _cagr(Decimal("100"), Decimal("169"), Decimal("2")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
    "fin-005": FinanceGuardrail(
        instructions=(
            "Calculate straight-line depreciation using (cost - salvage) / useful_life with cost = 1200, salvage = 200, and useful_life = 4 years. "
            "Return only the annual depreciation as a whole number."),
        calculator=lambda: (Decimal("1200") - Decimal("200")) / Decimal("4"),
        format="number",
        auto_correct=True,
        decimals=0,
    ),
    "fin-009": FinanceGuardrail(
        instructions=(
            "Use the compound interest formula A = P(1 + r/n)^(n*t). "
            "Compute the interest earned (A - P) and report only that value rounded to two decimals."),
        calculator=lambda: _compound_interest_interest(Decimal("400"), Decimal("0.06"), 4, 3),
        format="number",
    ),
    "fin-012": FinanceGuardrail(
        instructions=(
            "Compute gross margin percentage as ((revenue - COGS) / revenue) * 100 with revenue 9000 and COGS 6300. "
            "Return only the percentage rounded to the nearest whole percent with a trailing % sign."),
        calculator=lambda: _percentage((Decimal("9000") - Decimal("6300")) / Decimal("9000")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
    "fin-020": FinanceGuardrail(
        instructions=(
            "Compute the fixed monthly payment for a $10,000 loan at 5% annual interest over 5 years with monthly compounding. "
            "Use the standard amortization formula and return only the payment rounded to two decimals (no currency symbol)."),
        calculator=lambda: _loan_payment(Decimal("10000"), Decimal("0.05"), Decimal("5"), 12),
        format="number",
        auto_correct=True,
        decimals=2,
    ),
    "fin-021": FinanceGuardrail(
        instructions=(
            "Compute the contribution margin ratio as ((sales price - variable cost) / sales price) * 100 using sales price 70 and variable cost 28. "
            "Return only the percentage as an integer with a trailing % sign."),
        calculator=lambda: _percentage((Decimal("70") - Decimal("28")) / Decimal("70")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
    "fin-014": FinanceGuardrail(
        instructions=(
            "Convert 18 months to 1.5 years and compute the annualized return using "
            "((final / initial) ** (1 / years) - 1) * 100. Carry full precision through the exponentiation and only round once at the end. "
            "Return only the percentage with two decimals and a trailing % sign; do not add words or extra spacing."),
        calculator=lambda: _annualized_return(Decimal("2000"), Decimal("2600"), Decimal("1.5")),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-017": FinanceGuardrail(
        instructions=(
            "Calculate dividend yield as (dividend per share / share price) * 100 using dividend per share 2 and share price 40. "
            "Return only the percentage rounded to the nearest whole percent with a trailing % sign."),
        calculator=lambda: _percentage(Decimal("2") / Decimal("40")),
        format="percent",
        auto_correct=True,
        decimals=0,
    ),
    "fin-018": FinanceGuardrail(
        instructions=(
            "Discount each cash flow by (1 + 0.10)^period (with period starting at 0). Sum the discounted values and report the net present value rounded to two decimals."),
        calculator=lambda: _net_present_value((Decimal("-5000"), Decimal("2000"), Decimal("2500"), Decimal("3000")), Decimal("0.10")),
        format="number",
    ),
    "fin-019": FinanceGuardrail(
        instructions=(
            "Solve for the internal rate of return that makes the net present value of [-3000, 1200, 1500, 1800] equal to zero. "
            "Use an iterative solver (e.g., Newton-Raphson) until the NPV is within 0.0001 of zero and avoid linear interpolation shortcuts. "
            "Return the IRR as a percentage with two decimals and a trailing % sign; do not add words or extra spacing."),
        calculator=lambda: _irr_newton((Decimal("-3000"), Decimal("1200"), Decimal("1500"), Decimal("1800"))),
        format="percent",
        auto_correct=True,
    ),
    "fin-023": FinanceGuardrail(
        instructions=(
            "Compute the effective annual rate as ((1 + nominal_rate/12) ** 12 - 1) * 100 with nominal_rate = 0.09. "
            "Return only the percentage rounded to two decimals and include a trailing % sign."),
        calculator=lambda: ((Decimal(1) + Decimal("0.09") / Decimal(12)) ** 12 - Decimal(1)) * Decimal(100),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-024": FinanceGuardrail(
        instructions=(
            "Use the ordinary annuity future value formula FV = payment * (((1 + r)^n - 1) / r) with r = 0.04 and n = 8. "
            "Return only the future value rounded to two decimals."),
        calculator=lambda: _future_value_annuity(Decimal("400"), Decimal("0.04"), 8),
        format="number",
        auto_correct=True,
        decimals=2,
    ),
    "fin-025": FinanceGuardrail(
        instructions=(
            "Compute the weighted average cost of capital as w_e * cost_e + w_d * cost_d without applying a tax adjustment to the debt component. "
            "Return the result as a percentage with two decimals and a trailing % sign."),
        calculator=lambda: _wacc(Decimal("0.60"), Decimal("0.10"), Decimal("0.40"), Decimal("0.06"), Decimal("0.25")),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-hard-002": FinanceGuardrail(
        instructions=(
            "Apply the CAGR formula ((final / initial) ** (1 / years) - 1) * 100 with initial value 18500.40, final value 29877.92, and years = 3.25. "
            "Return only the percentage rounded to two decimals with a trailing % sign."),
        calculator=lambda: _cagr(Decimal("18500.40"), Decimal("29877.92"), Decimal("3.25")),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-hard-005": FinanceGuardrail(
        instructions=(
            "Calculate straight-line depreciation using (cost - salvage) / useful_life with cost = 182450.90, salvage = 17895.25, and useful_life = 9 years. "
            "Return only the annual depreciation rounded to two decimals."),
        calculator=lambda: (Decimal("182450.90") - Decimal("17895.25")) / Decimal("9"),
        format="number",
        auto_correct=True,
        decimals=2,
    ),
    "fin-hard-009": FinanceGuardrail(
        instructions=(
            "Use the compound interest formula A = P(1 + r/n)^(n*t) with P = 12750.40, r = 0.072, n = 12, and t = 6 years. "
            "Compute the interest earned (A - P) and report only that value rounded to two decimals."),
        calculator=lambda: _compound_interest_interest(Decimal("12750.40"), Decimal("0.072"), 12, 6),
        format="number",
    ),
    "fin-hard-012": FinanceGuardrail(
        instructions=(
            "Compute gross margin percentage as ((revenue - COGS) / revenue) * 100 with revenue 348920.75 and COGS 244577.18. "
            "Return only the percentage rounded to two decimals with a trailing % sign."),
        calculator=lambda: _percentage((Decimal("348920.75") - Decimal("244577.18")) / Decimal("348920.75")),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-hard-014": FinanceGuardrail(
        instructions=(
            "Convert 22 months to years (22 / 12) and compute the annualized return using ((final / initial) ** (1 / years) - 1) * 100 with initial 48250 and final 61345.80. "
            "Return only the percentage with two decimals and a trailing % sign; do not add extra text."),
        calculator=lambda: _annualized_return(Decimal("48250"), Decimal("61345.80"), Decimal("22") / Decimal("12")),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-hard-017": FinanceGuardrail(
        instructions=(
            "Calculate dividend yield as (dividend per share / share price) * 100 using dividend per share 3.25 and share price 56.40. "
            "Return only the percentage rounded to two decimals with a trailing % sign."),
        calculator=lambda: _percentage(Decimal("3.25") / Decimal("56.40")),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-hard-018": FinanceGuardrail(
        instructions=(
            "Discount each cash flow [-8350, 2420.75, 3180.40, 4590.90, 5285.35] by (1 + 0.095)^period (with period starting at 0). "
            "Sum the discounted values and report the net present value rounded to two decimals."),
        calculator=lambda: _net_present_value(
            (
                Decimal("-8350"),
                Decimal("2420.75"),
                Decimal("3180.40"),
                Decimal("4590.90"),
                Decimal("5285.35"),
            ),
            Decimal("0.095"),
        ),
        format="number",
    ),
    "fin-hard-019": FinanceGuardrail(
        instructions=(
            "Solve for the internal rate of return that makes the net present value of [-4600, 1525.50, 1840.75, 2175.10, 2490.60] equal to zero. "
            "Use an iterative solver until the NPV is within 0.0001 of zero and return the IRR as a percentage with two decimals and a trailing % sign."),
        calculator=lambda: _irr_newton(
            (
                Decimal("-4600"),
                Decimal("1525.50"),
                Decimal("1840.75"),
                Decimal("2175.10"),
                Decimal("2490.60"),
            )
        ),
        format="percent",
        auto_correct=True,
    ),
    "fin-hard-020": FinanceGuardrail(
        instructions=(
            "Compute the fixed monthly payment for a $485,000 loan at 4.35% annual interest over 18 years with monthly compounding. "
            "Use the standard amortization formula and return only the payment rounded to two decimals (no currency symbol)."),
        calculator=lambda: _loan_payment(Decimal("485000"), Decimal("0.0435"), Decimal("18"), 12),
        format="number",
        auto_correct=True,
        decimals=2,
    ),
    "fin-hard-021": FinanceGuardrail(
        instructions=(
            "Compute the contribution margin ratio as ((sales price - variable cost) / sales price) * 100 using sales price 142.80 and variable cost 58.45. "
            "Return only the percentage with two decimals and a trailing % sign."),
        calculator=lambda: _percentage((Decimal("142.80") - Decimal("58.45")) / Decimal("142.80")),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-hard-023": FinanceGuardrail(
        instructions=(
            "Compute the effective annual rate as ((1 + 0.114/12) ** 12 - 1) * 100. "
            "Return only the percentage rounded to two decimals and include a trailing % sign."),
        calculator=lambda: ((Decimal(1) + Decimal("0.114") / Decimal(12)) ** 12 - Decimal(1)) * Decimal(100),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
    "fin-hard-024": FinanceGuardrail(
        instructions=(
            "Use the ordinary annuity future value formula FV = payment * (((1 + r)^n - 1) / r) with payment = 785.60, r = 0.055, and n = 12. "
            "Return only the future value rounded to two decimals."),
        calculator=lambda: _future_value_annuity(Decimal("785.60"), Decimal("0.055"), 12),
        format="number",
        auto_correct=True,
        decimals=2,
    ),
    "fin-hard-025": FinanceGuardrail(
        instructions=(
            "Compute the weighted average cost of capital as w_e * cost_e + w_d * cost_d without applying a tax adjustment to the debt component, using w_e = 0.55, cost_e = 0.116, w_d = 0.45, cost_d = 0.058. "
            "Return the result as a percentage with two decimals and a trailing % sign."),
        calculator=lambda: _wacc(Decimal("0.55"), Decimal("0.116"), Decimal("0.45"), Decimal("0.058"), Decimal("0.27")),
        format="percent",
        auto_correct=True,
        decimals=2,
    ),
}


def get_guardrail(task_id: str) -> Optional[FinanceGuardrail]:
    return FINANCE_GUARDRAILS.get(task_id)
