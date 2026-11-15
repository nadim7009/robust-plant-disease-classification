"""Per-class metrics, confusion matrices and reliability bins from the saved test logits (CPU only).

Writes OUT_DIR/extra/:
  clean_metrics.csv        per run: accuracy, macro precision/recall/F1, weighted F1, test loss, mean confidence
  clean_metrics_mean.csv   the same as mean and SD per configuration
  per_class_<config>.csv   per-class precision/recall/F1 (mean over seeds) and test support
  confusion_pairs_<config>.csv  most frequent (true, predicted) pairs, mean errors per seed
  confusion_<config>_seed0.npy  38 x 38 confusion matrix of seed 0
  reliability_<config>_clean_seed0.csv  bin table (count, mean confidence, accuracy) for the reliability diagram
"""
import json, os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from run_experiments import load_splits

ROOT = Path(os.environ.get("OUT_DIR", "/kaggle/working/revision"))
OUT = ROOT / "extra"
OUT.mkdir(parents=True, exist_ok=True)
classes = sorted(load_splits()["test"].class_name.unique())
assert len(classes) == 38


def softmax(z):
    z = z.astype(np.float64)
    p = np.exp(z - z.max(1, keepdims=True))
    return p / p.sum(1, keepdims=True)


rows, per_class, conf = [], {}, {}
for f in sorted((ROOT / "results").glob("*.json")):
    r = json.load(open(f))
    cfg = f"{r['arch']}_{r['variant']}" + ("_os" if r["oversample"] else "")
    z = np.load(ROOT / "logits" / f"{r['run']}.npz")
    y = z["labels"]
    p = softmax(z["clean"])
    pred = p.argmax(1)
    P, R, F, S = precision_recall_fscore_support(y, pred, labels=range(38), zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y, pred, labels=range(38), average="weighted", zero_division=0)
    rows.append({"config": cfg, "seed": r["seed"], "acc": 100 * (pred == y).mean(),
                 "macro_precision": 100 * P.mean(), "macro_recall": 100 * R.mean(), "macro_f1": 100 * F.mean(),
                 "weighted_f1": 100 * wf, "test_loss": float(-np.log(p[np.arange(len(y)), y] + 1e-12).mean()),
                 "mean_conf": 100 * p.max(1).mean(), "errors": int((pred != y).sum())})
    per_class.setdefault(cfg, []).append(pd.DataFrame({"class": classes, "precision": P, "recall": R, "f1": F, "support": S}))
    cm = confusion_matrix(y, pred, labels=range(38))
    conf.setdefault(cfg, []).append(cm)
    if r["seed"] == 0:
        np.save(OUT / f"confusion_{cfg}_seed0.npy", cm)
        edges = np.linspace(0, 1, 11)
        b = np.clip(np.digitize(p.max(1), edges[1:-1]), 0, 9)
        pd.DataFrame([{"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "count": int((b == i).sum()),
                       "mean_conf": 100 * p.max(1)[b == i].mean() if (b == i).any() else np.nan,
                       "acc": 100 * (pred == y)[b == i].mean() if (b == i).any() else np.nan} for i in range(10)]
                     ).to_csv(OUT / f"reliability_{cfg}_clean_seed0.csv", index=False)

df = pd.DataFrame(rows)
df.round(4).to_csv(OUT / "clean_metrics.csv", index=False)
g = df.drop(columns="seed").groupby("config")
pd.concat({"mean": g.mean(), "sd": g.std()}, axis=1).round(4).to_csv(OUT / "clean_metrics_mean.csv")
print(g.mean().round(2).to_string())

for cfg, tabs in per_class.items():
    t = pd.concat(tabs).groupby("class", sort=False).agg(precision=("precision", "mean"), recall=("recall", "mean"),
                                                         f1=("f1", "mean"), f1_sd=("f1", "std"), support=("support", "first"))
    t.round(4).to_csv(OUT / f"per_class_{cfg}.csv")
    cm = np.mean(conf[cfg], axis=0)
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(-cm.ravel())[:10], cm.shape))[0]
    pd.DataFrame([{"true": classes[i], "pred": classes[j], "mean_errors_per_seed": round(float(cm[i, j]), 1)} for i, j in idx]
                 ).to_csv(OUT / f"confusion_pairs_{cfg}.csv", index=False)

e = per_class["efficientnet_v2_s_A0"]
print(pd.read_csv(OUT / "per_class_efficientnet_v2_s_A0.csv").sort_values("f1").head(10).to_string(index=False))
print(pd.read_csv(OUT / "confusion_pairs_efficientnet_v2_s_A0.csv").to_string(index=False))
print("written to", OUT)
