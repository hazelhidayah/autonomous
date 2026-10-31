"""Train the collision-risk models.

Two layers:
  a) regression of `risk` (log10 Pc) with a Ridge baseline and a
     HistGradientBoosting main model,
  b) tier classification, three ways: rules applied to predicted risk,
     a direct 3-class model, and the max-severity union of the two.

Validation is GroupKFold by event_id so rows from one event never span
train and validation. Probabilities are isotonic-calibrated on
out-of-fold predictions.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GroupKFold

from common import TIERS, assign_tier, engineer, fit_imputation_values, load

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(exist_ok=True)
RNG = 42
N_FOLDS = 5

TIER_TO_INT = {t: i for i, t in enumerate(TIERS)}  # Safe=0, Elevated=1, Critical=2


def make_regressor():
    return HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        random_state=RNG,
    )


def make_classifier():
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.08,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        random_state=RNG,
    )


def balanced_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(TIERS))
    w = len(y) / (len(TIERS) * counts)
    return w[y]


def ridge_matrix(X: pd.DataFrame) -> np.ndarray:
    """Numeric-only, median-imputed, standardised matrix for the baseline."""
    cols = [
        "log_miss_distance",
        "log_t_sigma_r",
        "log_c_sigma_r",
        "log_miss_over_sigma",
        "relative_speed",
        "time_to_tca",
        "max_risk_estimate",
    ]
    M = X[cols].astype(float)
    M = M.fillna(M.median())
    return ((M - M.mean()) / M.std()).to_numpy()


def cls_report(y_true, y_pred, label):
    rep = classification_report(
        y_true, y_pred, target_names=TIERS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    fn_critical = int(sum(cm[2]) - cm[2][2])  # true Critical predicted lower
    print(f"\n=== {label} ===  (false-negative Critical: {fn_critical})")
    print(classification_report(y_true, y_pred, target_names=TIERS, zero_division=0))
    return {"report": rep, "confusion_matrix": cm, "fn_critical": fn_critical}


def band_metrics(y_true, y_pred, lo=-8.0, hi=-3.0):
    """Regression error inside the band that decides tiers."""
    m = (y_true >= lo) & (y_true <= hi)
    if m.sum() == 0:
        return None
    return {
        "n": int(m.sum()),
        "mae": mean_absolute_error(y_true[m], y_pred[m]),
        "rmse": float(np.sqrt(mean_squared_error(y_true[m], y_pred[m]))),
    }


def main():
    train = load("train")
    test = load("test")

    imputation_values = fit_imputation_values(train)
    X = engineer(train, imputation_values)
    y_risk = train["risk"].to_numpy()
    y_tier = pd.Series(assign_tier(y_risk, train["miss_distance"])).map(TIER_TO_INT).to_numpy()
    groups = train["event_id"].to_numpy()

    X_test = engineer(test, imputation_values)
    yt_risk = test["risk"].to_numpy()
    yt_tier = (
        pd.Series(assign_tier(yt_risk, test["miss_distance"])).map(TIER_TO_INT).to_numpy()
    )

    gkf = GroupKFold(n_splits=N_FOLDS)
    oof_risk = np.full(len(train), np.nan)
    oof_risk_ridge = np.full(len(train), np.nan)
    oof_proba = np.full((len(train), 3), np.nan)

    for fold, (tr, va) in enumerate(gkf.split(X, y_risk, groups)):
        reg = make_regressor().fit(X.iloc[tr], y_risk[tr])
        oof_risk[va] = reg.predict(X.iloc[va])

        ridge = Ridge(alpha=1.0).fit(ridge_matrix(X)[tr], y_risk[tr])
        oof_risk_ridge[va] = ridge.predict(ridge_matrix(X)[va])

        clf = make_classifier().fit(
            X.iloc[tr], y_tier[tr], sample_weight=balanced_weights(y_tier[tr])
        )
        oof_proba[va] = clf.predict_proba(X.iloc[va])
        print(f"fold {fold} done")

    metrics = {}

    # --- regression ---
    metrics["regression"] = {
        "ridge_baseline": {
            "mae": mean_absolute_error(y_risk, oof_risk_ridge),
            "rmse": float(np.sqrt(mean_squared_error(y_risk, oof_risk_ridge))),
            "band_-8_-3": band_metrics(y_risk, oof_risk_ridge),
        },
        "hgb": {
            "mae": mean_absolute_error(y_risk, oof_risk),
            "rmse": float(np.sqrt(mean_squared_error(y_risk, oof_risk))),
            "band_-8_-3": band_metrics(y_risk, oof_risk),
        },
    }
    # On the rows that matter (true risk above the -30 floor)
    live = y_risk > -30
    metrics["regression"]["hgb_above_floor"] = {
        "mae": mean_absolute_error(y_risk[live], oof_risk[live]),
        "rmse": float(np.sqrt(mean_squared_error(y_risk[live], oof_risk[live]))),
    }
    print("\nRegression (group-CV OOF):")
    for name in ("ridge_baseline", "hgb"):
        r = metrics["regression"][name]
        print(f"  {name:15s} MAE {r['mae']:.2f}  RMSE {r['rmse']:.2f}  "
              f"band[-8,-3] MAE {r['band_-8_-3']['mae']:.2f} (n={r['band_-8_-3']['n']})")

    # --- calibration (isotonic, one-vs-rest on OOF, renormalised) ---
    isos = []
    cal_proba = np.zeros_like(oof_proba)
    for k in range(3):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
        iso.fit(oof_proba[:, k], (y_tier == k).astype(float))
        cal_proba[:, k] = iso.predict(oof_proba[:, k])
        isos.append(iso)
    cal_proba /= cal_proba.sum(axis=1, keepdims=True)
    metrics["calibration"] = {
        "log_loss_raw": log_loss(y_tier, oof_proba, labels=[0, 1, 2]),
        "log_loss_calibrated": log_loss(y_tier, cal_proba, labels=[0, 1, 2]),
    }

    # --- three classification strategies (out-of-fold) ---
    pred_rules = (
        pd.Series(assign_tier(oof_risk, train["miss_distance"])).map(TIER_TO_INT).to_numpy()
    )
    pred_direct = cal_proba.argmax(axis=1)
    pred_union = np.maximum(pred_rules, pred_direct)

    metrics["cv_classification"] = {
        "rules_on_predicted_risk": cls_report(y_tier, pred_rules, "CV: rules on predicted risk"),
        "direct_classifier": cls_report(y_tier, pred_direct, "CV: direct 3-class"),
        "max_severity_union": cls_report(y_tier, pred_union, "CV: max-severity union"),
    }

    def crit_recall(block):
        return block["report"]["Critical"]["recall"]

    best_name = max(metrics["cv_classification"], key=lambda k: crit_recall(metrics["cv_classification"][k]))
    metrics["selected_strategy"] = best_name
    print(f"\nSelected strategy (best Critical recall): {best_name}")

    # --- test-mimicking holdout: OOF rows with time_to_tca >= 2 days ---
    from config import TEST_MIN_TCA_DAYS

    far = (train["time_to_tca"] >= TEST_MIN_TCA_DAYS).to_numpy()
    strategy_oof = {"rules_on_predicted_risk": pred_rules,
                    "direct_classifier": pred_direct,
                    "max_severity_union": pred_union}[best_name]
    metrics["cv_tca_ge_2"] = cls_report(
        y_tier[far], strategy_oof[far],
        f"CV restricted to time_to_tca >= {TEST_MIN_TCA_DAYS} ({best_name})",
    )
    metrics["cv_tca_ge_2"]["reg_mae"] = mean_absolute_error(y_risk[far], oof_risk[far])

    # --- leakage sanity check: shuffled labels should be near chance ---
    shuffled = np.random.default_rng(RNG).permutation(y_tier)
    tr0, va0 = next(gkf.split(X, shuffled, groups))
    shuffled_clf = make_classifier().fit(
        X.iloc[tr0], shuffled[tr0], sample_weight=balanced_weights(shuffled[tr0])
    )
    shuffled_pred = shuffled_clf.predict(X.iloc[va0])
    shuffled_report = classification_report(
        y_tier[va0], shuffled_pred, target_names=TIERS, output_dict=True, zero_division=0
    )
    metrics["shuffled_label_sanity"] = {
        "fold_rows": int(len(va0)),
        "macro_f1": float(shuffled_report["macro avg"]["f1-score"]),
        "critical_recall": float(shuffled_report["Critical"]["recall"]),
        "expected_macro_f1_baseline": 1 / len(TIERS),
    }

    # --- error slices and concrete failure cases ---
    selected_oof = strategy_oof
    errors = train[["c_object_type", "time_to_tca", "risk", "miss_distance"]].copy()
    errors["true_tier"] = y_tier
    errors["predicted_tier"] = selected_oof
    errors["risk_band"] = pd.cut(
        errors["risk"], [-np.inf, -10, -8, -6, -4, -3, np.inf],
        labels=["< -10", "-10 to -8", "-8 to -6", "-6 to -4", "-4 to -3", ">= -3"],
    )
    errors["tca_bucket"] = pd.cut(
        errors["time_to_tca"], [-np.inf, 1, 2, 7, np.inf],
        labels=["<1 day", "1-2 days", "2-7 days", ">7 days"],
    )
    def slice_report(group):
        rep = classification_report(
            group["true_tier"], group["predicted_tier"],
            labels=[0, 1, 2], target_names=TIERS, output_dict=True, zero_division=0,
        )
        return {"n": int(len(group)), "macro_f1": float(rep["macro avg"]["f1-score"]),
                "critical_recall": float(rep["Critical"]["recall"])}

    metrics["error_breakdown"] = {
        "object_type": {str(k): slice_report(g) for k, g in errors.groupby(
            "c_object_type", dropna=False, observed=True
        )},
        "time_to_tca": {str(k): slice_report(g) for k, g in errors.groupby(
            "tca_bucket", observed=True
        )},
        "risk_band": {str(k): slice_report(g) for k, g in errors.groupby(
            "risk_band", observed=True
        )},
    }
    failures = errors[(errors["true_tier"] == 2) & (errors["predicted_tier"] < 2)]
    metrics["failure_cases"] = {
        "false_negative_critical": int(len(failures)),
        "worst_misses": errors.nsmallest(10, "miss_distance")[
            ["miss_distance", "risk", "true_tier", "predicted_tier"]
        ].to_dict("records"),
        "false_negative_examples": failures.nlargest(10, "risk")[
            ["miss_distance", "risk", "true_tier", "predicted_tier"]
        ].to_dict("records"),
    }

    # --- permutation importance on the last fold's validation split ---
    reg_full = make_regressor().fit(X.iloc[tr], y_risk[tr])
    pi = permutation_importance(
        reg_full, X.iloc[va], y_risk[va], n_repeats=8, random_state=RNG, n_jobs=-1
    )
    importance = sorted(
        zip(X.columns, pi.importances_mean, pi.importances_std),
        key=lambda t: -t[1],
    )
    metrics["permutation_importance"] = [
        {"feature": f, "mean": float(m), "std": float(s)} for f, m, s in importance
    ]

    # --- final fit on all train, predict test ---
    reg = make_regressor().fit(X, y_risk)
    clf = make_classifier().fit(X, y_tier, sample_weight=balanced_weights(y_tier))

    test_risk = reg.predict(X_test)
    test_proba = clf.predict_proba(X_test)
    test_cal = np.column_stack([isos[k].predict(test_proba[:, k]) for k in range(3)])
    test_cal /= test_cal.sum(axis=1, keepdims=True)

    t_rules = (
        pd.Series(assign_tier(test_risk, test["miss_distance"])).map(TIER_TO_INT).to_numpy()
    )
    t_direct = test_cal.argmax(axis=1)
    t_union = np.maximum(t_rules, t_direct)
    strategy_preds = {
        "rules_on_predicted_risk": t_rules,
        "direct_classifier": t_direct,
        "max_severity_union": t_union,
    }
    test_pred_tier = strategy_preds[best_name]

    metrics["test_regression"] = {
        "mae": mean_absolute_error(yt_risk, test_risk),
        "rmse": float(np.sqrt(mean_squared_error(yt_risk, test_risk))),
        "band_-8_-3": band_metrics(yt_risk, test_risk),
    }
    metrics["test_classification"] = {
        name: cls_report(yt_tier, pred, f"TEST: {name}")
        for name, pred in strategy_preds.items()
    }

    # --- outputs ---
    int_to_tier = {v: k for k, v in TIER_TO_INT.items()}
    preds = pd.DataFrame(
        {
            "event_id": test["event_id"],
            "time_to_tca": test["time_to_tca"],
            "predicted_risk": test_risk,
            "predicted_tier": pd.Series(test_pred_tier).map(int_to_tier),
            "p_safe": test_cal[:, 0],
            "p_elevated": test_cal[:, 1],
            "p_critical": test_cal[:, 2],
        }
    )
    preds.to_csv(OUT / "test_predictions.csv", index=False)

    # Spec-format deliverable: event_id, row index, prediction columns.
    spec = preds.copy()
    spec.insert(1, "row_index", np.arange(len(spec)))
    spec = spec.drop(columns=["time_to_tca"])
    spec.to_csv(OUT / "predictions.csv", index=False)

    oof = pd.DataFrame(
        {
            "event_id": train["event_id"],
            "time_to_tca": train["time_to_tca"],
            "predicted_risk": oof_risk,
            "predicted_tier": pd.Series(np.maximum(pred_rules, pred_direct) if best_name == "max_severity_union" else (pred_rules if best_name == "rules_on_predicted_risk" else pred_direct)).map(int_to_tier),
            "p_safe": cal_proba[:, 0],
            "p_elevated": cal_proba[:, 1],
            "p_critical": cal_proba[:, 2],
        }
    )
    oof.to_csv(OUT / "oof_predictions.csv", index=False)

    with open(OUT / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("\nWrote outputs/test_predictions.csv, outputs/oof_predictions.csv, outputs/metrics.json")


if __name__ == "__main__":
    main()
