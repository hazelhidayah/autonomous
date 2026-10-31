"""Fit the final models on all training data and persist them for the app."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

from common import TIERS, assign_tier, engineer, fit_imputation_values, load
from train import TIER_TO_INT, balanced_weights, make_classifier, make_regressor

OUT = Path(__file__).resolve().parents[1] / "outputs"

train = load("train")
imputation_values = fit_imputation_values(train)
X = engineer(train, imputation_values)
y_risk = train["risk"].to_numpy()
y_tier = pd.Series(assign_tier(y_risk, train["miss_distance"])).map(TIER_TO_INT).to_numpy()
groups = train["event_id"].to_numpy()

# isotonic calibrators must be fit on out-of-fold probabilities
gkf = GroupKFold(n_splits=5)
oof_proba = np.full((len(train), 3), np.nan)
for tr, va in gkf.split(X, y_tier, groups):
    clf = make_classifier().fit(X.iloc[tr], y_tier[tr], sample_weight=balanced_weights(y_tier[tr]))
    oof_proba[va] = clf.predict_proba(X.iloc[va])

isos = []
for k in range(3):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    iso.fit(oof_proba[:, k], (y_tier == k).astype(float))
    isos.append(iso)

reg = make_regressor().fit(X, y_risk)
clf = make_classifier().fit(X, y_tier, sample_weight=balanced_weights(y_tier))

joblib.dump(
    {"regressor": reg, "classifier": clf, "isotonics": isos,
     "imputation_values": imputation_values,
     "category_levels": list(X["c_object_type"].cat.categories)},
    OUT / "models.joblib",
)
print("saved", OUT / "models.joblib")
