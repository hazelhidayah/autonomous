"""Single place for thresholds, paths, seeds and feature lists."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = (
    ROOT / "data" / "Collision Avoidance Challenge - Dataset" / "kelvins_competition_data"
)
OUT_DIR = ROOT / "outputs"
DB_PATH = ROOT / "data" / "ops.db"
MODELS_PATH = OUT_DIR / "models.joblib"

SEED = 42
N_FOLDS = 5

# Tier thresholds (risk = log10 Pc; miss distance in metres). The operational
# framing follows NASA/SP-20205011318 and ESA Space Debris Office practice;
# these values are challenge thresholds, not a substitute for full Pc analysis.
RISK_CRITICAL = -4.0
RISK_ELEVATED = -6.0
MISS_CRITICAL_M = 1_000.0
MISS_ELEVATED_M = 5_000.0

TIERS = ["Safe", "Elevated", "Critical"]

# Test contains only rows this far from closest approach; used for the
# test-mimicking holdout report.
TEST_MIN_TCA_DAYS = 2.0

# Raw columns allowed as model inputs (risk is the target, never a feature).
BASE_FEATURES = [
    "miss_distance",
    "relative_speed",
    "relative_position_r",
    "relative_position_t",
    "relative_position_n",
    "c_sigma_r",
    "t_sigma_r",
    "time_to_tca",
    "F10",
    "AP",
    "max_risk_estimate",
]
CATEGORICAL = ["c_object_type"]

# Alerting: notifications beyond the dashboard are off by default.
ALERT_WEBHOOK_URL: str | None = None
