"""
Grad-CAM on a fixed random sample of test images, using a checkpoint saved by
run_experiments.py.

    python gradcam.py --run resnet50_A0_seed0 --n_correct 50 --n_wrong 50
    python gradcam.py --run resnet50_A4_seed0 --n_correct 50 --n_wrong 50 --segmented_root /path/to/segmented

Writes grid figures (300 dpi) and samples.csv to OUT_DIR/gradcam/<run>/.
With --segmented_root (the PlantVillage "segmented" images, black background)
it also measures the share of the Grad-CAM map that falls on the leaf.
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from run_experiments import OUT, build_model, load_splits, make_transforms


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.act, self.grad = model, None, None
        self.h1 = layer.register_forward_hook(lambda m, i, o: setattr(self, "act", o.detach()))
        self.h2 = layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grad", go[0].detach()))

    def __call__(self, x, cls):
        self.model.zero_grad()
        out = self.model(x)
        out[:, cls].sum().backward()
        cam = F.relu((self.grad.mean((2, 3), keepdim=True) * self.act).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze()
        return ((cam - cam.min()) / (cam.max() - cam.min() + 1e-8)).cpu().numpy()


def target_layer(model, arch):
    if arch == "resnet50":
        return model.layer4
    if arch == "densenet121":
        return model.features
    return model.features[-1]  # EfficientNet-B0, EfficientNetV2-S, MobileNetV3


def seg_key(name):
    # "abc.JPG" and "abc_final_masked.jpg" map to the same key
    stem = Path(name).stem.lower()
    return stem[:-len("_final_masked")] if stem.endswith("_final_masked") else stem


def leaf_mask(path, seg_lookup):
    seg = seg_lookup.get(seg_key(path))
    if seg is None:
        return None
    return np.asarray(Image.open(seg).convert("L").resize((224, 224))) > 10  # black background in the segmented set


def pretty(c):
    # "Tomato___Spider_mites Two-spotted_spider_mite" -> two short lines that fit above a 224 px panel
    return textwrap.fill(c.replace("___", ": ").replace("_", " "), 26)


def grid(items, title, path):
    rows = len(items)
    fig, axes = plt.subplots(rows, 2, figsize=(5.0, 2.1 * rows))  # 4 rows = 8.4 in, fits one journal page
    axes = np.atleast_2d(axes)
    for ax_row, it in zip(axes, items):
        ax_row[0].imshow(it["img"]); ax_row[1].imshow(it["img"]); ax_row[1].imshow(it["cam"], cmap="jet", alpha=0.45)
        ax_row[0].set_title(f"True:\n{pretty(it['true'])}", fontsize=8)
        ax_row[1].set_title(f"Pred ({it['conf']:.1f}%):\n{pretty(it['pred'])}", fontsize=8)
        for a in ax_row:
            a.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="e.g. resnet50_A0_seed0")
    ap.add_argument("--n_correct", type=int, default=50)
    ap.add_argument("--n_wrong", type=int, default=50)
    ap.add_argument("--segmented_root", default=None)
    a = ap.parse_args()

    arch = next(x for x in ["efficientnet_v2_s", "efficientnet_b0", "mobilenet_v3_large", "densenet121", "resnet50"]
                if a.run.startswith(x))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 100 images run fine on a CPU
    model, _ = build_model(arch)
    model.load_state_dict(torch.load(OUT / "checkpoints" / f"{a.run}.pth", map_location="cpu"))
    model.to(device).eval()
    for p in model.parameters():  # backbone was frozen in training; Grad-CAM needs gradients at the target layer
        p.requires_grad_(True)
    cam_fn = GradCAM(model, target_layer(model, arch))

    test = load_splits()["test"]
    _, eval_tf = make_transforms([], False)
    z = np.load(OUT / "logits" / f"{a.run}.npz")
    logits, labels = z["clean"].astype(np.float32), z["labels"]
    assert len(labels) == len(test) and (labels == test.label.values).all(), "logits do not match test order"
    probs = torch.softmax(torch.tensor(logits), 1).numpy()
    pred = probs.argmax(1)

    rng = np.random.default_rng(0)  # fixed sample, listed in samples.csv
    correct_idx = rng.choice(np.where(pred == labels)[0], a.n_correct, replace=False)
    wrong_pool = np.where(pred != labels)[0]
    wrong_idx = rng.choice(wrong_pool, min(a.n_wrong, len(wrong_pool)), replace=False)
    classes = sorted(test.class_name.unique())

    seg_lookup = {}
    if a.segmented_root:
        seg_lookup = {seg_key(p.name): str(p) for d in Path(a.segmented_root).iterdir() if d.is_dir() for p in d.iterdir()}
        print(len(seg_lookup), "segmented images indexed")

    out_dir = OUT / "gradcam" / a.run
    out_dir.mkdir(parents=True, exist_ok=True)
    records, shown = [], {"correct": [], "wrong": []}
    for kind, idx in (("correct", correct_idx), ("wrong", wrong_idx)):
        for i in idx:
            img = Image.open(test.path[i]).convert("RGB")
            cam = cam_fn(eval_tf(img).unsqueeze(0).to(device), int(pred[i]))
            rec = {"index": int(i), "kind": kind, "true": classes[labels[i]], "pred": classes[pred[i]],
                   "conf": 100 * probs[i].max()}
            mask = leaf_mask(test.path[i], seg_lookup) if seg_lookup else None
            if mask is not None:
                rec["cam_on_leaf"] = float((cam * mask).sum() / (cam.sum() + 1e-8))
                rec["leaf_area"] = float(mask.mean())  # share expected if the map ignored the leaf
            records.append(rec)
            if len(shown[kind]) < 60:
                shown[kind].append({**rec, "img": np.asarray(img.resize((224, 224))), "cam": cam})
    df = pd.DataFrame(records)
    df.to_csv(out_dir / "samples.csv", index=False)

    for kind, items in shown.items():
        for page in range(0, len(items), 4):  # 4 images per figure keeps 8 pt labels on one page
            grid(items[page:page + 4], f"{a.run}: {kind} predictions",
                 out_dir / f"{kind}_{page // 4 + 1:02d}.png")
    if "cam_on_leaf" in df:
        df["off_leaf_majority"] = df.cam_on_leaf < 0.5
        s = df.groupby("kind").agg(n=("cam_on_leaf", "count"), cam_on_leaf_mean=("cam_on_leaf", "mean"),
                                   cam_on_leaf_median=("cam_on_leaf", "median"), leaf_area_mean=("leaf_area", "mean"),
                                   share_maps_mostly_off_leaf=("off_leaf_majority", "mean"))
        s.round(4).to_csv(out_dir / "cam_on_leaf_summary.csv")
        print(s.round(3).to_string())
        print("matched masks:", df.cam_on_leaf.notna().sum(), "of", len(df))
    print(f"written to {out_dir}")
