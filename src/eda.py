"""Quick EDA: verify units, distributions and tier balance before modelling."""

import json

import numpy as np
import pandas as pd

from common import TIERS, assign_tier, load

pd.set_option("display.width", 200)

train = load("train")
test = load("test")

report = {}
report["train_rows"] = len(train)
report["train_events"] = train["event_id"].nunique()
report["test_rows"] = len(test)
report["test_events"] = test["event_id"].nunique()
report["cdms_per_event_median"] = float(train.groupby("event_id").size().median())

r = train["risk"]
report["risk"] = {
    "min": float(r.min()),
    "max": float(r.max()),
    "median": float(r.median()),
    "pct_at_floor_-30": float((r <= -30).mean() * 100),
    "pct_ge_-6": float((r >= -6).mean() * 100),
    "pct_ge_-4": float((r >= -4).mean() * 100),
}

md = train["miss_distance"]
report["miss_distance"] = {
    "min": float(md.min()),
    "median": float(md.median()),
    "max": float(md.max()),
    "units_check": "median tens of km if metres",
}

report["time_to_tca"] = {
    "train_min": float(train["time_to_tca"].min()),
    "train_max": float(train["time_to_tca"].max()),
    "test_min": float(test["time_to_tca"].min()),
    "test_max": float(test["time_to_tca"].max()),
}

for name, df in [("train", train), ("test", test)]:
    tiers = pd.Series(assign_tier(df["risk"], df["miss_distance"]))
    report[f"{name}_tier_counts"] = tiers.value_counts().reindex(TIERS).to_dict()

# Missingness in the allowed features
from common import BASE_FEATURES, CATEGORICAL

miss = train[BASE_FEATURES + CATEGORICAL].isna().mean() * 100
report["feature_missing_pct"] = {k: round(float(v), 2) for k, v in miss.items() if v > 0}

# Is max_risk_estimate leaky? Correlation with risk, and availability.
sub = train[["risk", "max_risk_estimate"]].dropna()
report["corr_risk_vs_max_risk_estimate"] = float(
    np.corrcoef(sub["risk"], sub["max_risk_estimate"])[0, 1]
)

report["c_object_type_values"] = train["c_object_type"].value_counts(dropna=False).to_dict()

print(json.dumps(report, indent=2, default=str))
