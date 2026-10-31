"""Permutation feature importance plot for the risk regressor."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
m = json.loads((ROOT / "outputs" / "metrics.json").read_text())
rows = m["permutation_importance"][:12][::-1]

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
names = [r["feature"] for r in rows]
vals = [r["mean"] for r in rows]
errs = [r["std"] for r in rows]
ax.barh(names, vals, xerr=errs, color="#2a78d6", height=0.55,
        error_kw={"ecolor": "#898781", "capsize": 2})
ax.set_xlabel("permutation importance (increase in MSE of log₁₀ Pc)",
              color="#52514e")
ax.set_title("What drives the predicted collision probability",
             color="#0b0b0b", loc="left", fontsize=12, pad=12)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")
ax.tick_params(colors="#52514e")
ax.xaxis.grid(True, color="#e1e0d9", linewidth=0.7)
ax.set_axisbelow(True)
plt.tight_layout()
out = ROOT / "outputs" / "feature_importance.png"
plt.savefig(out, facecolor="#fcfcfb")
print(f"wrote {out}")
