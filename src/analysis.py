"""
Summary tables, paired tests and calibration from the outputs of run_experiments.py.
No training: it reads results/*.json and logits/*.npz under OUT_DIR.

    python analysis.py            # writes OUT_DIR/analysis/

Outputs
  summary.csv / summary_table.tex        mean ± SD over seeds per configuration
  per_condition.csv / per_condition.tex  accuracy of every configuration under every condition
  paired_tests.csv                       paired t-tests against a reference, Holm-corrected
  calibration.csv / calibration.tex      ECE, MCE and Brier score, clean and corrupted
  reliability_*.png                      reliability diagrams, 300 dpi
  compute.csv                            training time and peak GPU memory
"""
import json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(os.environ.get("OUT_DIR", "/kaggle/working/revision"))
OUT = ROOT / "analysis"
N_BINS = 10  # equal-width confidence bins
CONDITIONS = ["clean"] + [f"{f}_{s}" for f in ("brightness", "gaussian_noise", "gaussian_blur")
                          for s in ("mild", "moderate", "severe")]
METRICS = ["clean_acc", "clean_macro_f1", "mean_corr_acc", "mean_corr_macro_f1", "worst_acc", "worst_macro_f1"]


def config_name(r):
    return f"{r['arch']}_{r['variant']}" + ("_os" if r["oversample"] else "")


def load_runs():
    rows = []
    for f in sorted((ROOT / "results").glob("*.json")):
        r = json.load(open(f))
        if r.get("smoke"):
            continue
        row = {"config": config_name(r), "run": r["run"], "seed": r["seed"],
               "clean_acc": r["results"]["clean"]["acc"], "clean_macro_f1": r["results"]["clean"]["macro_f1"],
               **{k: r[k] for k in METRICS[2:]},
               "train_min": r["train_seconds"] / 60, "peak_gpu_mb": r["peak_gpu_mem_mb"],
               "epochs": r["epochs_trained"], "best_epoch": r["best_epoch"]}
        row.update({f"acc_{c}": r["results"][c]["acc"] for c in CONDITIONS})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- calibration
def calibration(logits, labels):
    z = logits.astype(np.float64)
    p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1)
    correct = (pred == labels).astype(float)
    edges = np.linspace(0, 1, N_BINS + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, N_BINS - 1)  # last bin includes 1.0
    ece, mce, bins = 0.0, 0.0, []
    for b in range(N_BINS):
        m = idx == b
        if m.any():
            gap = abs(correct[m].mean() - conf[m].mean())
            ece += m.mean() * gap; mce = max(mce, gap)
            bins.append((conf[m].mean(), correct[m].mean(), int(m.sum())))
    onehot = np.eye(p.shape[1])[labels]
    brier = ((p - onehot) ** 2).sum(1).mean()
    return {"ece": 100 * ece, "mce": 100 * mce, "brier": brier, "mean_conf": 100 * conf.mean(),
            "acc": 100 * correct.mean()}, bins


def calibration_table(runs):
    rows = []
    for _, r in runs.iterrows():
        z = np.load(ROOT / "logits" / f"{r['run']}.npz")
        for c in CONDITIONS:
            m, _ = calibration(z[c], z["labels"])
            rows.append({"config": r["config"], "seed": r["seed"], "condition": c, **m})
    return pd.DataFrame(rows)


def reliability_plot(run_a, run_b, condition, path):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.4), sharey=True)
    for ax, run in zip(axes, (run_a, run_b)):
        z = np.load(ROOT / "logits" / f"{run}.npz")
        m, bins = calibration(z[condition], z["labels"])
        conf, acc, n = zip(*bins)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.bar(conf, acc, width=0.08, edgecolor="black", alpha=0.7)
        ax.set_title(f"{run.split('_seed')[0]}\nECE {m['ece']:.2f}%, MCE {m['mce']:.2f}%", fontsize=9)
        ax.set_xlabel("Mean confidence", fontsize=9); ax.tick_params(labelsize=8)
    axes[0].set_ylabel("Accuracy", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


# ---------------------------------------------------------------- statistics
def holm(pvals):
    order = np.argsort(pvals); adj = np.empty(len(pvals)); running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i]); adj[i] = min(1.0, running)
    return adj


def paired_tests(runs, pairs, metrics):
    rows = []
    for ref, other in pairs:
        a = runs[runs.config == ref].set_index("seed"); b = runs[runs.config == other].set_index("seed")
        seeds = a.index.intersection(b.index)
        if len(seeds) < 3:
            continue
        for m in metrics:
            d = b.loc[seeds, m] - a.loc[seeds, m]
            t, p = stats.ttest_rel(b.loc[seeds, m], a.loc[seeds, m])
            half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
            rows.append({"reference": ref, "compared": other, "metric": m, "n_seeds": len(seeds),
                         "mean_diff_pp": d.mean(), "ci95_low": d.mean() - half, "ci95_high": d.mean() + half,
                         "t": t, "p": p})
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = holm(np.nan_to_num(df.p.values, nan=1.0))  # nan only if all seeds tie
        df["sig"] = np.where(df.p_holm < 0.05, "*", "n.s.")
    return df


# ---------------------------------------------------------------- tables
def fmt(m, s, nd):
    # SD is undefined with one seed: print the mean alone instead of NaN
    return m.apply(lambda c: c.map(lambda v: f"{v:.{nd}f}")).where(s.isna(),
           m.apply(lambda c: c.map(lambda v: f"{v:.{nd}f}")) + " ± " + s.apply(lambda c: c.map(lambda v: f"{v:.{nd}f}")))


def mean_sd(df, cols):
    g = df.groupby("config")[cols]
    return fmt(g.mean(), g.std(ddof=1), 2), g.size()


def booktabs(df, caption, path):
    cols = "l" + "c" * len(df.columns)
    body = df.to_latex(escape=True, column_format=cols)  # pandas emits \toprule/\midrule/\bottomrule (three-line table)
    path.write_text(f"% {caption}\n% needs \\usepackage{{booktabs}}\n" + body)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    assert len(runs), "no finished runs found"
    print(runs.groupby("config").size().rename("seeds").to_string())

    summary, n = mean_sd(runs, METRICS)
    summary.insert(0, "seeds", n)
    summary.to_csv(OUT / "summary.csv"); booktabs(summary, "Mean ± SD over seeds", OUT / "summary_table.tex")

    per_cond, _ = mean_sd(runs, [f"acc_{c}" for c in CONDITIONS])
    per_cond.columns = CONDITIONS
    per_cond.to_csv(OUT / "per_condition.csv"); booktabs(per_cond, "Accuracy per condition", OUT / "per_condition.tex")

    runs.groupby("config")[["train_min", "peak_gpu_mb", "epochs", "best_epoch"]].agg(["mean", "std"]).round(2) \
        .to_csv(OUT / "compute.csv")

    # Ablation vs A0, oversampling vs its base, architectures vs ResNet50
    pairs = [("resnet50_A0", f"resnet50_A{k}") for k in range(1, 5)]
    pairs += [("resnet50_A0", "resnet50_A0_os"), ("resnet50_A4", "resnet50_A4_os")]
    pairs += [("resnet50_A1", "resnet50_A4")]  # best single corruption vs. the combined policy
    pairs += [(f"{a}_A0", f"{a}_A0_os") for a in ("densenet121", "efficientnet_b0", "efficientnet_v2_s")]
    pairs += [("resnet50_A0", f"{a}_A0") for a in ("densenet121", "efficientnet_b0", "efficientnet_v2_s", "mobilenet_v3_large")]
    tests = paired_tests(runs, pairs, METRICS)
    tests.round(4).to_csv(OUT / "paired_tests.csv", index=False)
    if len(tests):
        print(tests[["reference", "compared", "metric", "mean_diff_pp", "p_holm", "sig"]].round(3).to_string(index=False))

    cal = calibration_table(runs)
    cal.to_csv(OUT / "calibration_all.csv", index=False)
    g = cal.groupby(["config", "condition"])[["ece", "mce", "brier"]]
    cal_summary = fmt(g.mean(), g.std(ddof=1), 3)
    cal_summary.to_csv(OUT / "calibration.csv")
    booktabs(cal_summary.loc[cal_summary.index.get_level_values(1).isin(
        ["clean", "gaussian_noise_severe", "gaussian_blur_severe", "brightness_severe"])],
        "Calibration, clean and severe corruptions", OUT / "calibration.tex")

    for cond in ("clean", "gaussian_blur_severe", "gaussian_noise_severe"):
        if {"resnet50_A0_seed0", "resnet50_A4_seed0"} <= set(runs.run):
            reliability_plot("resnet50_A0_seed0", "resnet50_A4_seed0", cond, OUT / f"reliability_A0_vs_A4_{cond}.png")
    print(f"written to {OUT}")
