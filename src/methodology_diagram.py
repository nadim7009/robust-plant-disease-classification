"""Fig. 1 of the paper, the methodology diagram.
Drawn at print size (6.6 in wide) so all text is at least 8 pt.

    python methodology_diagram.py fig01_methodology.png
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "fig01_methodology.png"
WI, HI = 6.6, 5.1
fig = plt.figure(figsize=(WI, HI))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, WI); ax.set_ylim(0, HI); ax.axis("off")
F, S = "DejaVu Sans", 8
LH = 0.145  # line height (in) at 8 pt
C = {"blue": ("#dce8f7", "#5b7fb5"), "split": ("#e8f0fa", "#5b7fb5"), "yellow": ("#fdf0cf", "#d4ac3a"), "orange": ("#fae3d6", "#d97b3d"),
     "purple": ("#e9e1f2", "#8a63b8"), "green": ("#e3f1df", "#5f9e4f")}
LIGHT = {"yellow": ("#fef8e8", "#e6cf8f"), "orange": ("#fdf0ea", "#efb894"), "purple": ("#f4eff9", "#c2aedb"), "green": ("#eef7eb", "#a9d29c")}


def rbox(x, y, w, h, fc, ec, lw=1.2, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", fc=fc, ec=ec, lw=lw, ls=ls))


def centred(x, y, w, h, colour, title, lines):
    rbox(x, y, w, h, *C[colour])
    yy = y + h - 0.09
    for t in title.split("\n"):
        ax.text(x + w / 2, yy, t, ha="center", va="top", fontsize=S + 0.5, fontweight="bold", family=F); yy -= LH + 0.01
    yy -= 0.05
    for ln in lines:
        if ln == "":
            yy -= 0.06; continue
        ax.text(x + w / 2, yy, ln.strip("*"), ha="center", va="top", fontsize=S, family=F, fontweight="bold" if ln.startswith("**") else "normal")
        yy -= LH


def listing(x, y, w, h, colour, title, items):
    rbox(x, y, w, h, *LIGHT[colour], lw=1.0)
    ax.text(x + 0.07, y + h - 0.08, title, ha="left", va="top", fontsize=S, fontweight="bold", family=F)
    for k, it in enumerate(items):
        ax.text(x + 0.07, y + h - 0.08 - (k + 1) * LH - 0.02, "• " + it, ha="left", va="top", fontsize=S, family=F)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color="#444", lw=1.0, shrinkA=0, shrinkB=0, mutation_scale=9))


def line(xs, ys):
    ax.plot(xs, ys, color="#444", lw=1.0, solid_capstyle="butt")


# columns: baseline | [corruptions  architectures] | ablation
BW = 1.42
XB, XC, XA, XL = 0.24, 1.86, 3.38, 5.1
TOP_MAIN, H_MAIN = 3.72, 1.2
TOP_SUB, H_SUB = 2.42, 0.84

centred(2.0, 4.62, 2.6, 0.42, "blue", "PlantVillage RGB colour set", ["54,305 images • 38 classes • 14 crops"])
arrow(3.3, 4.62, 3.3, 4.48)
centred(1.05, 3.93, 4.5, 0.55, "split", "Official train/test split + leaf-group integrity check",
        ["Train 39,207  |  Validation 4,389 (group-aware)  |  Test 10,709", "**Every run: seeds 0–4, at most 10 epochs, patience 3"])

# dashed group around corruptions + architectures
rbox(XC - 0.08, TOP_SUB - H_SUB - 0.08, (XA + BW) - XC + 0.16, TOP_MAIN - (TOP_SUB - H_SUB) + 0.16, "#f6f7f9", "#aaaaaa", lw=0.9, ls="--")

# edges from the split box
line([1.05, XB + BW / 2], [4.2, 4.2]); arrow(XB + BW / 2, 4.2, XB + BW / 2, TOP_MAIN)
line([3.3, 3.3], [3.93, 3.86]); line([XC + BW / 2, XA + BW / 2], [3.86, 3.86])
arrow(XC + BW / 2, 3.86, XC + BW / 2, TOP_MAIN); arrow(XA + BW / 2, 3.86, XA + BW / 2, TOP_MAIN)

y0 = TOP_MAIN - H_MAIN
centred(XB, y0, BW, H_MAIN, "yellow", "Clean Baseline", ["EfficientNetV2-S", "Frozen backbone", "Linear head", "", "**Clean test", "**evaluation"])
centred(XC, y0, BW, H_MAIN, "orange", "Controlled\nCorruptions", ["Brightness", "Gaussian noise", "Gaussian blur", "", "**3 × 3 = 9 conditions"])
centred(XA, y0, BW, H_MAIN, "purple", "Architecture\nComparison", ["ResNet50", "DenseNet121", "EfficientNet-B0", "MobileNetV3-L", "**Clean + corrupted"])
centred(XL, y0, BW, H_MAIN, "green", "Corruption-Aware\nTraining Ablation", ["ResNet50", "A0 none | A1 blur", "A2 noise | A3 bright.", "A4 all three", "**+ OS for A0, A4"])

# baseline -> corruptions, group -> ablation
arrow(XB + BW, y0 + H_MAIN / 2, XC, y0 + H_MAIN / 2)
gx = XA + BW + 0.08  # right edge of the dashed group
line([gx, gx + 0.07], [y0 - 0.25, y0 - 0.25]); line([gx + 0.07, gx + 0.07], [y0 - 0.25, y0 + H_MAIN / 2]); arrow(gx + 0.07, y0 + H_MAIN / 2, XL, y0 + H_MAIN / 2)

ys = TOP_SUB - H_SUB
listing(XB, ys, BW, H_SUB, "yellow", "Baseline Setup", ["Resize to 224 px", "ImageNet mean/SD", "38-class head", "Clean metrics"])
listing(XC, ys, BW, H_SUB, "orange", "Robustness Tests", ["At 256 × 256 px", "Severity 1 / 2 / 3", "Same test set", "ImageNet-C map"])
listing(XA, ys, BW, H_SUB, "purple", "Model Comparison", ["Same split, budget", "Seeds 0–4", "vs ResNet50", "Time, GPU memory"])
listing(XL, ys, BW, H_SUB, "green", "Training Ablation", ["A1–A4 vs A0", "A4 vs A1", "OS vs base", "Distinct images"])

# bottom panel: analyses for every model, fed from the baseline's left side
PB, PH = 0.1, 1.28
rbox(0.38, PB, 6.0, PH, "#f6f7f9", "#aaaaaa", lw=0.9, ls="--")
ax.text(3.38, PB + PH - 0.07, "Analyses for every model and seed (clean and corrupted test sets)", ha="center", va="top", fontsize=S + 0.5, fontweight="bold", family=F)
bw, bh, by = 1.8, 0.92, PB + 0.1
centred(0.55, by, bw, bh, "blue", "Calibration Analysis", ["ECE and MCE (10 bins)", "Multiclass Brier score", "Reliability diagrams"])
centred(2.48, by, bw, bh, "orange", "Error and Statistics", ["Per-class P / R / F1", "Confusion matrix", "Paired t-tests, Holm"])
centred(4.41, by, bw, bh, "purple", "Explainability Analysis", ["Grad-CAM, ResNet50 A0/A4", "100 random test images", "Share of activation on leaf"])
line([XB, 0.12], [y0 + H_MAIN / 2, y0 + H_MAIN / 2]); line([0.12, 0.12], [y0 + H_MAIN / 2, PB + PH / 2]); arrow(0.12, PB + PH / 2, 0.38, PB + PH / 2)

fig.savefig(OUT, dpi=300, facecolor="white")
print("written", OUT)
