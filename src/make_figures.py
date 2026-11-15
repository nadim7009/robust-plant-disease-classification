"""Figures 5 to 18 of the paper from the five-seed results (300 dpi, text at least 8 pt).

Some file numbers differ from the paper: fig15_* is Fig. 16 and fig16_* is Fig. 15."""
import glob, json, os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(os.environ.get("OUT_DIR", "/kaggle/working/revision"))  # holds results/, analysis/, extra/, gradcam/
OUT = Path(os.environ.get("FIG_DIR", R / "figures")); OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.titlesize": 8, "axes.labelsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "savefig.dpi": 300})
COND = ["clean"] + [f"{f}_{s}" for f in ("brightness", "gaussian_noise", "gaussian_blur") for s in ("mild", "moderate", "severe")]
FAM = {"brightness": "Brightness reduction", "gaussian_noise": "Gaussian noise", "gaussian_blur": "Gaussian blur"}
SEV = ["mild", "moderate", "severe"]

runs = []
for f in glob.glob(str(R / "results/*.json")):
    r = json.load(open(f))
    cfg = f"{r['arch']}_{r['variant']}" + ("_os" if r["oversample"] else "")
    row = {"config": cfg, "seed": r["seed"], "history": r["history"]}
    for c in COND:
        row[f"acc_{c}"] = r["results"][c]["acc"]; row[f"f1_{c}"] = r["results"][c]["macro_f1"]
    row["mean_corr"] = r["mean_corr_acc"]
    runs.append(row)
df = pd.DataFrame(runs)
m = df.drop(columns="history").groupby("config").mean(numeric_only=True)
s = df.drop(columns="history").groupby("config").std(numeric_only=True)
ARCH = [("resnet50_A0", "ResNet50"), ("densenet121_A0", "DenseNet121"), ("efficientnet_b0_A0", "EfficientNet-B0"),
        ("efficientnet_v2_s_A0", "EfficientNetV2-S"), ("mobilenet_v3_large_A0", "MobileNetV3-L")]
ABL = [(f"resnet50_A{i}", n) for i, n in enumerate(["A0", "A1 blur", "A2 noise", "A3 bright.", "A4 all"])]
GREY = "#4C72B0"


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=300); plt.close(fig)


# Fig 5: EfficientNetV2-S validation accuracy and macro F1 per epoch, five seeds
fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.2))
for _, r in df[df.config == "efficientnet_v2_s_A0"].sort_values("seed").iterrows():
    e = [h["epoch"] for h in r.history]
    ax[0].plot(e, [h["acc"] for h in r.history], marker="o", ms=2.5, lw=1, label=f"seed {r.seed}")
    ax[1].plot(e, [h["macro_f1"] for h in r.history], marker="o", ms=2.5, lw=1)
ax[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)"); ax[1].set(xlabel="Epoch", ylabel="Validation macro F1 (%)")
ax[0].legend(frameon=False, ncol=1)
for a in ax: a.set_xticks(range(1, 11)); a.grid(alpha=.3)
save(fig, "fig05_effv2s_val_curves.png")

# Fig 6/7: architectures, clean and mean corrupted accuracy (mean +- SD)
for name, col, ylab, lo in (("fig06_arch_clean.png", "acc_clean", "Clean test accuracy (%)", 85),
                            ("fig07_arch_meancorr.png", "mean_corr", "Mean corrupted acc. (%)", 50)):
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    x = np.arange(len(ARCH))
    ax.bar(x, [m.loc[c, col] for c, _ in ARCH], yerr=[s.loc[c, col] for c, _ in ARCH], color=GREY, capsize=3, width=.6)
    for i, (c, _) in enumerate(ARCH):
        ax.text(i, m.loc[c, col] + s.loc[c, col] + .4, f"{m.loc[c, col]:.1f}", ha="center", va="bottom")
    ax.set_xticks(x, [n for _, n in ARCH], rotation=30, ha="right"); ax.set_ylabel(ylab); ax.set_ylim(lo, 100)
    ax.grid(axis="y", alpha=.3)
    save(fig, name)


def severity(cfg, key, fname, ylab):
    fig, ax = plt.subplots(figsize=(3.0, 2.2))
    xs = [0, 1, 2, 3]
    for fam, lab in FAM.items():
        ys = [m.loc[cfg, f"{key}_clean"]] + [m.loc[cfg, f"{key}_{fam}_{v}"] for v in SEV]
        es = [s.loc[cfg, f"{key}_clean"]] + [s.loc[cfg, f"{key}_{fam}_{v}"] for v in SEV]
        ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, capsize=2, lw=1, label=lab)
    ax.axhline(m.loc[cfg, f"{key}_clean"], ls="--", c="grey", lw=.8)
    ax.set_xticks(xs, ["Clean", "Mild", "Moderate", "Severe"]); ax.set_ylabel(ylab); ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="lower left"); ax.grid(alpha=.3)
    save(fig, fname)


severity("efficientnet_v2_s_A0", "acc", "fig08_effv2s_acc_severity.png", "Test accuracy (%)")
severity("efficientnet_v2_s_A0", "f1", "fig09_effv2s_f1_severity.png", "Macro F1 (%)")
severity("resnet50_A0", "acc", "fig10_resnet50_A0_severity.png", "Test accuracy (%)")

# Fig 11: A0-A4 accuracy per condition (grouped bars)
fig, ax = plt.subplots(figsize=(6.2, 2.5))
labels = ["Clean"] + [f"{lab.split()[-1].capitalize() if fam != 'brightness' else 'Bright.'}\n{v}" for fam, lab in FAM.items() for v in SEV]
x = np.arange(len(COND)); w = .16
for i, (c, n) in enumerate(ABL):
    ax.bar(x + (i - 2) * w, [m.loc[c, f"acc_{k}"] for k in COND], w, yerr=[s.loc[c, f"acc_{k}"] for k in COND],
           capsize=1, error_kw={"lw": .6}, label=n)
ax.set_xticks(x, labels); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.legend(frameon=False, ncol=5, loc="lower left", bbox_to_anchor=(0, 1.0)); ax.grid(axis="y", alpha=.3)
save(fig, "fig11_ablation_conditions.png")

# Fig 12: clean vs mean corrupted accuracy, ablation variants and oversampling
fig, ax = plt.subplots(figsize=(3.0, 2.4))
pts = ABL + [("resnet50_A0_os", "A0 + OS"), ("resnet50_A4_os", "A4 + OS")]
for c, n in pts:
    ax.errorbar(m.loc[c, "acc_clean"], m.loc[c, "mean_corr"], xerr=s.loc[c, "acc_clean"], yerr=s.loc[c, "mean_corr"],
                fmt="o", ms=3.5, capsize=2, lw=.8)
    off = {"A4 all": (6, 4), "A4 + OS": (-38, -11), "A0": (-14, 5), "A0 + OS": (6, -10), "A3 bright.": (6, 2), "A2 noise": (-44, 3), "A1 blur": (6, 3)}[n]
    ax.annotate(n, (m.loc[c, "acc_clean"], m.loc[c, "mean_corr"]), textcoords="offset points", xytext=off)
ax.set(xlabel="Clean accuracy (%)", ylabel="Mean corrupted accuracy (%)"); ax.grid(alpha=.3)
ax.set_xlim(95.6, 97.4)
save(fig, "fig12_clean_vs_corr.png")

# Fig 13: A0 vs A4 by family and severity
fig, ax = plt.subplots(1, 3, figsize=(6.2, 2.0), sharey=True)
for a, (fam, lab) in zip(ax, FAM.items()):
    for c, n, st in (("resnet50_A0", "A0", "--"), ("resnet50_A4", "A4", "-")):
        ys = [m.loc[c, "acc_clean"]] + [m.loc[c, f"acc_{fam}_{v}"] for v in SEV]
        es = [s.loc[c, "acc_clean"]] + [s.loc[c, f"acc_{fam}_{v}"] for v in SEV]
        a.errorbar(range(4), ys, yerr=es, ls=st, marker="o", ms=3, capsize=2, lw=1, label=n)
    a.set_title(lab); a.set_xticks(range(4), ["Clean", "Mild", "Mod.", "Severe"]); a.grid(alpha=.3); a.set_ylim(0, 100)
ax[0].set_ylabel("Accuracy (%)"); ax[0].legend(frameon=False, loc="lower left")
save(fig, "fig13_A0_vs_A4_family.png")

m.to_csv(OUT / "means.csv"); s.to_csv(OUT / "sds.csv")
print("ok", sorted(p.name for p in OUT.glob("*.png")))


# ---- figures that need the logits and the output of extra_stats.py
def bins(logits, labels, n=10):
    z = logits.astype(np.float64); p = np.exp(z - z.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    conf, pred = p.max(1), p.argmax(1); b = np.clip(np.digitize(conf, np.linspace(0, 1, n + 1)[1:-1]), 0, n - 1)
    rows = [(np.nan, np.nan, 0) if not (b == i).any() else (conf[b == i].mean(), (pred == labels)[b == i].mean(), int((b == i).sum())) for i in range(n)]
    ece = sum(c * abs(a - m) for m, a, c in rows if c) / len(labels)
    return rows, 100 * ece


def rel_axis(ax, rows, title):
    centers = (np.arange(10) + .5) / 10
    ax.bar(centers, [a if c else 0 for _, a, c in rows], width=.09, color=GREY, edgecolor="k", lw=.4)
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(title)
    ax.set_xlabel("Mean confidence")


EX = R / "extra"
if (EX / "reliability_efficientnet_v2_s_A0_clean_seed0.csv").exists():
    t = pd.read_csv(EX / "reliability_efficientnet_v2_s_A0_clean_seed0.csv")
    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    rel_axis(ax, [(mc / 100, a / 100, c) for mc, a, c in zip(t.mean_conf, t.acc, t["count"])], "")
    ax.set_ylabel("Accuracy")
    save(fig, "fig14_effv2s_reliability_clean.png")

    cm = np.load(EX / "confusion_efficientnet_v2_s_A0_seed0.npy").astype(float)
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(0, 38, 2), [str(i + 1) for i in range(0, 38, 2)]); ax.set_yticks(range(0, 38, 2), [str(i + 1) for i in range(0, 38, 2)])
    ax.set(xlabel="Predicted class index", ylabel="True class index")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04, label="Share of true class")
    save(fig, "fig15_effv2s_confusion_seed0.png")

if (R / "logits" / "resnet50_A0_seed0.npz").exists():
    fig, ax = plt.subplots(2, 3, figsize=(6.2, 4.3), sharex=True, sharey=True)
    for i, (run, name) in enumerate((("resnet50_A0_seed0", "A0"), ("resnet50_A4_seed0", "A4"))):
        z = np.load(R / "logits" / f"{run}.npz")
        for j, (cond, lab) in enumerate((("clean", "clean"), ("gaussian_blur_severe", "severe blur"), ("gaussian_noise_severe", "severe noise"))):
            rows, ece = bins(z[cond], z["labels"])
            rel_axis(ax[i, j], rows, f"{name}, {lab} (ECE {ece:.1f}%)")
        ax[i, 0].set_ylabel("Accuracy")
    for a in ax[0]: a.set_xlabel("")
    save(fig, "fig16_reliability_A0_A4.png")

for k in ("correct", "wrong"):  # Grad-CAM pages drawn by gradcam.py (4 images, 8 pt labels)
    src = R / "gradcam" / "resnet50_A4_seed0" / f"{k}_01.png"
    if src.exists():
        (OUT / f"fig{17 if k == 'correct' else 18}_gradcam_A4_{k}.png").write_bytes(src.read_bytes())
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
