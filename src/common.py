"""Shared data loading, labelling and feature engineering for the
collision-avoidance challenge (ESA Kelvins CDM dataset)."""

import numpy as np
import pandas as pd

from config import (
    BASE_FEATURES,
    CATEGORICAL,
    DATA_DIR,
    MISS_CRITICAL_M,
    MISS_ELEVATED_M,
    RISK_CRITICAL,
    RISK_ELEVATED,
    TIERS,
)

__all__ = [
    "BASE_FEATURES", "CATEGORICAL", "TIERS",
    "MISS_CRITICAL_M", "MISS_ELEVATED_M", "RISK_CRITICAL", "RISK_ELEVATED",
    "load", "assign_tier", "fit_imputation_values", "engineer", "tier_reason",
]


def load(split: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{split}_data.csv")


def assign_tier(risk: np.ndarray, miss_distance_m: np.ndarray) -> np.ndarray:
    """Map (log10 Pc, miss distance in metres) to a tier.

    Critical is checked first so no row can be downgraded by a later rule.
    Threshold framing: NASA/SP-20205011318 and ESA Space Debris Office practice.
    """
    risk = np.asarray(risk, dtype=float)
    miss = np.asarray(miss_distance_m, dtype=float)
    tier = np.full(risk.shape, "Safe", dtype=object)
    # Apply the most severe rule first and only classify the remaining rows.
    critical = (risk >= RISK_CRITICAL) | (miss < MISS_CRITICAL_M)
    tier[critical] = "Critical"
    elevated = (~critical) & ((risk >= RISK_ELEVATED) | (miss <= MISS_ELEVATED_M))
    tier[elevated] = "Elevated"
    return tier


def tier_reason(risk: float, miss_m: float) -> str:
    """Plain-English reason for a tier, for alerts and explanations."""
    parts = []
    if miss_m < MISS_CRITICAL_M:
        parts.append(f"miss distance {miss_m:,.0f} m < 1 km keep-out")
    if risk >= RISK_CRITICAL:
        parts.append(f"predicted log₁₀ Pc {risk:.1f} ≥ −4")
    if not parts:
        if miss_m <= MISS_ELEVATED_M:
            parts.append(f"miss distance {miss_m:,.0f} m inside 5 km")
        if risk >= RISK_ELEVATED:
            parts.append(f"predicted log₁₀ Pc {risk:.1f} ≥ −6")
    return "; ".join(parts) if parts else "below all thresholds"


def fit_imputation_values(df: pd.DataFrame) -> dict[str, float]:
    """Fit feature-only median imputers on a training frame.

    The medians are fitted without using ``risk`` and are persisted with the
    model so scoring new CSVs uses the same values as training.
    """
    return {
        column: float(pd.to_numeric(df[column], errors="coerce").median())
        for column in BASE_FEATURES
        if column != "c_object_type"
    }


def engineer(
    df: pd.DataFrame, imputation_values: dict[str, float] | None = None
) -> pd.DataFrame:
    """Build the model matrix. Only uses quantities available in the
    current CDM, so it stays valid row-by-row as messages arrive.

    Missing values are represented by indicator columns before median
    imputation. The medians are fitted on training features and reused for
    validation, test, and uploaded data.
    """
    values = imputation_values or fit_imputation_values(df)
    X = df[BASE_FEATURES].copy()
    for column, median in values.items():
        X[column] = pd.to_numeric(X[column], errors="coerce").fillna(median)
    X["c_object_type"] = df["c_object_type"].fillna("UNKNOWN").astype(str)

    rel_pos_mag = np.sqrt(
        X["relative_position_r"] ** 2
        + X["relative_position_t"] ** 2
        + X["relative_position_n"] ** 2
    )
    combined_sigma = np.sqrt(X["t_sigma_r"] ** 2 + X["c_sigma_r"] ** 2)
    miss = X["miss_distance"].clip(lower=1.0)

    X["rel_pos_mag"] = rel_pos_mag
    X["combined_sigma"] = combined_sigma
    X["log_miss_distance"] = np.log10(miss)
    X["log_t_sigma_r"] = np.log10(X["t_sigma_r"].clip(lower=1e-6))
    X["log_c_sigma_r"] = np.log10(X["c_sigma_r"].clip(lower=1e-6))
    # Rough Mahalanobis-style distance: the strongest physical driver of Pc.
    X["miss_over_sigma"] = X["miss_distance"] / combined_sigma.clip(lower=1e-6)
    X["log_miss_over_sigma"] = np.log10(X["miss_over_sigma"].clip(lower=1e-9))
    X["speed_x_time"] = X["relative_speed"] * X["time_to_tca"]
    # Encounter geometry direction (unit-ish components).
    X["rel_r_over_miss"] = X["relative_position_r"] / miss
    X["rel_t_over_miss"] = X["relative_position_t"] / miss
    X["rel_n_over_miss"] = X["relative_position_n"] / miss
    # Missing-indicator flags for the columns with real gaps.
    X["space_weather_missing"] = df["F10"].isna().astype(np.int8)
    X["c_sigma_r_missing"] = df["c_sigma_r"].isna().astype(np.int8)

    X["c_object_type"] = X["c_object_type"].astype("category")
    return X
