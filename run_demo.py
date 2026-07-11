"""
End-to-end pipeline demo:
  synthetic merchant -> train (temporal split) -> shadow-score -> verify labels -> Money Report
Run:  python run_demo.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.report import generate as make_report
from backend.score import mature_outcomes, score_pending
from backend.train import train
from demo.generate_synthetic import generate

MERCHANT = "modanova"

print("=" * 62)
print("ReturnML — end-to-end shadow-mode pipeline demo")
print("=" * 62)

print("\n[1/5] Generating synthetic merchant history (18 months)...")
generate(MERCHANT, n_carts=20000)

print("\n[2/5] Training per-merchant model (temporal split, no leakage)...")
metrics, version = train(MERCHANT)
print(f"  model version : {version}")
print(f"  validation AUC: {metrics['auc']:.3f}  "
      f"(train n={metrics['n_train']}, valid n={metrics['n_valid']})")
for name, op in metrics["operating_points"].items():
    print(f"  {name}: threshold={op['threshold']:.2f} "
          f"precision={op['precision']:.2f} recall={op['recall']:.2f}")

print("\n[3/5] Shadow-scoring all carts (async, no customer impact)...")
n = score_pending(MERCHANT)
print(f"  scored {n} carts silently")

print("\n[4/5] Checking label maturity...")
n_mature = mature_outcomes(MERCHANT)
print(f"  {n_mature} outcomes past maturity (return window + buffer)")

print("\n[5/5] Generating the Money Report...")
report = make_report(MERCHANT)

out = Path("reports") / f"{MERCHANT}_money_report.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(report, indent=2))

c = report["corpus"]
print(f"""
{'=' * 62}
MONEY REPORT — {MERCHANT}
{'=' * 62}
Carts scored & mature : {c['carts_scored_and_mature']:,}
Overall return rate   : {c['overall_return_rate']:.1%}
VERIFIED AUC          : {c['verified_auc']:.3f}
""")
for op in report["operating_points"]:
    print(f"  [{op['name'].upper():12s}] flag {op['carts_flagged']:,} carts "
          f"| precision {op['verified_precision']:.0%} "
          f"| catches {op['recall_of_all_returns']:.0%} of all returns")
    print(f"  {'':14s} identified return cost  EUR {op['identified_return_cost_eur']:>10,.0f}")
    print(f"  {'':14s} simulated coupon profit EUR {op['simulated_coupon_net_profit_eur']:>10,.0f}\n")

m = report["merchandising"]
print("  Bracketing effect  :", m["bracketing_return_rate"])
print("  By payment method  :", m["payment_return_rate"])
print("  Worst SKU          :", m["worst_skus"][0] if m["worst_skus"] else "n/a")
print(f"\nFull report -> {out}")
