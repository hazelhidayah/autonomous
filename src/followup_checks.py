"""Checks raised after the first training round:
1. CV metrics restricted to train rows with time_to_tca >= 2 d
   (the only regime the test set contains).
2. max_risk_estimate vs risk, specifically in the decision band.
3. Calibration on test, where the Critical share is higher than train.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, log_loss, mean_absolute_error

from common import TIERS, assign_tier, load

OUT = Path(__file__).resolve().parents[1] / "outputs"
TIER_TO_INT = {t: i for i, t in enumerate(TIERS)}

train = load("train")
test = load("test")
oof = pd.read_csv(OUT / "oof_predictions.csv")
tpr = pd.read_csv(OUT / "test_predictions.csv")

y_tier = pd.Series(assign_tier(train["risk"], train["miss_distance"])).map(TIER_TO_INT)
pred_tier = oof["predicted_tier"].map(TIER_TO_INT)

out = {}

# 1. test-comparable validation: train OOF rows with time_to_tca >= 2
far = train["time_to_tca"] >= 2.0
out["oof_rows_tca_ge_2"] = int(far.sum())
rep = classification_report(
    y_tier[far], pred_tier[far], target_names=TIERS, output_dict=True, zero_division=0
)
out["cv_tca_ge_2"] = {
    t: {k: round(rep[t][k], 3) for k in ("precision", "recall", "f1-score")} for t in TIERS
}
out["cv_tca_ge_2"]["macro_f1"] = round(rep["macro avg"]["f1-score"], 3)
out["cv_tca_ge_2_reg_mae"] = round(
    mean_absolute_error(train.loc[far, "risk"], oof.loc[far.to_numpy(), "predicted_risk"]), 3
)

# 2. max_risk_estimate in the decision band
sub = train[["risk", "max_risk_estimate"]].dropna()
band = sub[(sub["risk"] >= -10) & (sub["risk"] <= -4)]
out["corr_mre_all"] = round(float(np.corrcoef(sub["risk"], sub["max_risk_estimate"])[0, 1]), 3)
out["corr_mre_band_-10_-4"] = round(
    float(np.corrcoef(band["risk"], band["max_risk_estimate"])[0, 1]), 3
)
live = sub[sub["risk"] > -30]
out["corr_mre_above_floor"] = round(
    float(np.corrcoef(live["risk"], live["max_risk_estimate"])[0, 1]), 3
)

# 3. calibration on test
yt = pd.Series(assign_tier(test["risk"], test["miss_distance"])).map(TIER_TO_INT)
probs = tpr[["p_safe", "p_elevated", "p_critical"]].to_numpy()
out["test_log_loss_calibrated"] = round(log_loss(yt, probs, labels=[0, 1, 2]), 4)
out["test_mean_p_critical"] = round(float(probs[:, 2].mean()), 4)
out["test_actual_critical_share"] = round(float((yt == 2).mean()), 4)
out["test_mean_p_elevated"] = round(float(probs[:, 1].mean()), 4)
out["test_actual_elevated_share"] = round(float((yt == 1).mean()), 4)

# reliability of p_critical in deciles
q = pd.qcut(probs[:, 2], 10, duplicates="drop")
rel = pd.DataFrame({"p": probs[:, 2], "y": (yt == 2).astype(int)}).groupby(q, observed=True)
out["test_p_critical_reliability"] = [
    {"mean_p": round(float(g["p"].mean()), 4), "frac_critical": round(float(g["y"].mean()), 4),
     "n": int(len(g))}
    for _, g in rel
]

print(json.dumps(out, indent=2))
(OUT / "followup_checks.json").write_text(json.dumps(out, indent=2))
