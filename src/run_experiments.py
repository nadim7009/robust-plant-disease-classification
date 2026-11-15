"""
Training and evaluation for every configuration in the paper: the five
architectures, the A0-A4 ablation on ResNet50 and the oversampling runs.
All of them use the same budget (at most 10 epochs, patience 3).

Each run writes results/<run_name>.json and logits/<run_name>.npz. A run whose
JSON already exists is skipped, so an interrupted session can be restarted.

Examples:
    !python run_experiments.py --arch resnet50 --variant A0 --seeds 0 1 2 3 4
    !python run_experiments.py --arch resnet50 --variant A4 --seeds 0 1 2 3 4
    !python run_experiments.py --arch efficientnet_b0 --variant A0 --seeds 0 1 2 3 4
    !python run_experiments.py --arch resnet50 --variant A0 --seeds 0 --smoke
"""
import argparse, json, os, random, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

# ---------------------------------------------------------------- paths
# Defaults are the Kaggle input paths; set the environment variables to run elsewhere.
PLANTVILLAGE_ROOT = Path(os.environ.get("PV_ROOT", "/kaggle/input/datasets/mahmudnadim/robust-crop-disease-plantvillage/color"))
SPLIT_ROOT = Path(os.environ.get("SPLIT_ROOT", "/kaggle/input/datasets/mahmudnadim/robust-crop-disease-splits/splits-20260929T040001Z-1-001/splits"))
TRAIN_VAL_FILE = Path(os.environ.get("TRAIN_VAL_FILE", "/kaggle/input/datasets/mahmudnadim/robust-crop-disease-splits/train_validation_split.npz"))
OUT = Path(os.environ.get("OUT_DIR", "/kaggle/working/revision"))

# ---------------------------------------------------------------- fixed protocol (Table 1 in the paper)
IMAGE_SIZE, BATCH_SIZE, NUM_CLASSES = 224, 32, 38
LR, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE = 1e-3, 1e-4, 10, 3
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TEST_NOISE_SEED = 12345  # fixed, so every model sees the same noisy test images

TEST_CONDITIONS = [  # (family, severity, parameter)
    ("brightness", "mild", 0.80), ("brightness", "moderate", 0.60), ("brightness", "severe", 0.40),
    ("gaussian_noise", "mild", 0.03), ("gaussian_noise", "moderate", 0.07), ("gaussian_noise", "severe", 0.12),
    ("gaussian_blur", "mild", 1.0), ("gaussian_blur", "moderate", 2.0), ("gaussian_blur", "severe", 3.0),
]
VARIANTS = {  # training corruption families per ablation variant
    "A0": [], "A1": ["blur"], "A2": ["noise"], "A3": ["brightness"], "A4": ["blur", "noise", "brightness"],
}


# ---------------------------------------------------------------- reproducibility
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RecordingSampler(WeightedRandomSampler):
    """Keeps the indices drawn in the last epoch so the distinct images can be counted."""
    def __iter__(self):
        self.last = list(super().__iter__())
        return iter(self.last)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)


# ---------------------------------------------------------------- data: official PlantVillage split, group-aware validation indices
def load_splits():
    read = lambda f: [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
    train_all, test = read(SPLIT_ROOT / "color_train.txt"), read(SPLIT_ROOT / "color_test.txt")
    idx = np.load(TRAIN_VAL_FILE)
    lookup = {p.name: str(p) for d in PLANTVILLAGE_ROOT.iterdir() if d.is_dir() for p in d.iterdir()}

    def resolve(rel):
        name = Path(rel).name
        return lookup.get(name) or lookup[re.sub(r"\s+(\.[^.]+)$", r"\1", name)]

    rows = [(resolve(r), Path(r).parent.name, s)
            for s, paths in [("train", [train_all[i] for i in idx["train_idx"]]),
                             ("val", [train_all[i] for i in idx["val_idx"]]),
                             ("test", test)]
            for r in paths]
    df = pd.DataFrame(rows, columns=["path", "class_name", "split"])
    classes = sorted(df.class_name.unique())
    df["label"] = df.class_name.map({c: i for i, c in enumerate(classes)})
    sizes = df.split.value_counts()
    assert len(classes) == 38 and (sizes["train"], sizes["val"], sizes["test"]) == (39_207, 4_389, 10_709)
    return {s: df[df.split == s].reset_index(drop=True) for s in ["train", "val", "test"]}


class ImageDataset(Dataset):
    def __init__(self, df, transform, pre=None):
        self.df, self.transform, self.pre = df, transform, pre  # pre: corruption on the 256x256 original

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.df.path[i]).convert("RGB")
        if self.pre is not None:
            img = self.pre(img, i)
        return self.transform(img), int(self.df.label[i])


# ---------------------------------------------------------------- corruptions
def add_noise(img, std, rng):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.clip(arr + rng.normal(0.0, std, arr.shape), 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def test_corruption(family, value):
    if family == "brightness":
        return lambda img, i: ImageEnhance.Brightness(img).enhance(value)
    if family == "gaussian_blur":
        return lambda img, i: img.filter(ImageFilter.GaussianBlur(radius=value))
    # noise seeded by image index: identical corrupted test set for every model and seed (paired tests)
    return lambda img, i: add_noise(img, value, np.random.default_rng(TEST_NOISE_SEED + i))


def train_corruption(families):
    def apply(img):  # each family is applied with p = 0.5
        if "blur" in families and random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.4, 1.2)))
        if "noise" in families and random.random() < 0.5:
            img = add_noise(img, random.uniform(0.02, 0.06), np.random)
        if "brightness" in families and random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.65, 1.35))
        return img
    return apply


def make_transforms(families, corrupt_before_resize):
    base = [transforms.RandomHorizontalFlip(0.5), transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
    corrupt = [transforms.Lambda(train_corruption(families))] if families else []
    order = corrupt + [resize] if corrupt_before_resize else [resize] + corrupt
    evaluate = transforms.Compose([resize, transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(order + base), evaluate


# ---------------------------------------------------------------- models: frozen backbone, trainable head
def build_model(arch):
    head = lambda n: nn.Sequential(nn.Dropout(0.2), nn.Linear(n, NUM_CLASSES))
    if arch == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = head(m.fc.in_features); trainable = m.fc
    elif arch == "densenet121":
        m = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        m.classifier = head(m.classifier.in_features); trainable = m.classifier
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        m.classifier = head(m.classifier[1].in_features); trainable = m.classifier
    elif arch == "efficientnet_v2_s":
        m = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        m.classifier = head(m.classifier[1].in_features); trainable = m.classifier
    elif arch == "mobilenet_v3_large":
        # the whole classifier (960 -> 1280 -> 38) is trained, not only the last layer
        m = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, NUM_CLASSES); trainable = m.classifier
    else:
        raise ValueError(arch)
    for p in m.parameters():
        p.requires_grad = False
    for p in trainable.parameters():
        p.requires_grad = True
    return m, trainable


# ---------------------------------------------------------------- train / evaluate
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device, non_blocking=True)).float().cpu()); labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


def scores(logits, labels):
    pred = logits.argmax(1)
    return {"acc": 100 * accuracy_score(labels, pred), "macro_f1": 100 * f1_score(labels, pred, average="macro", zero_division=0)}


def run(arch, variant, seed, oversample, corrupt_before_resize, data, device, smoke):
    name = f"{arch}_{variant}{'_os' if oversample else ''}_seed{seed}"
    res_path, logit_path = OUT / "results" / f"{name}.json", OUT / "logits" / f"{name}.npz"
    if res_path.exists():
        print(f"skip {name} (done)"); return
    set_seed(seed)
    train_df, val_df, test_df = data["train"], data["val"], data["test"]
    if smoke:  # quick pipeline check only, numbers are meaningless
        train_df, val_df, test_df = (d.sample(500, random_state=0).reset_index(drop=True) for d in (train_df, val_df, test_df))

    train_tf, eval_tf = make_transforms(VARIANTS[variant], corrupt_before_resize)
    g = torch.Generator().manual_seed(seed)
    loader_kw = dict(batch_size=BATCH_SIZE, num_workers=min(4, os.cpu_count() or 2), pin_memory=True, worker_init_fn=seed_worker)
    if oversample:  # weight = 1 / sqrt(class proportion)
        prop = train_df.label.map(train_df.label.value_counts(normalize=True)).values
        w = 1.0 / np.sqrt(prop); w = w / w.mean()
        sampler = RecordingSampler(torch.tensor(w, dtype=torch.double), len(train_df), replacement=True, generator=g)
        train_loader = DataLoader(ImageDataset(train_df, train_tf), sampler=sampler, generator=g, **loader_kw)
    else:
        train_loader = DataLoader(ImageDataset(train_df, train_tf), shuffle=True, generator=g, **loader_kw)
    val_loader = DataLoader(ImageDataset(val_df, eval_tf), shuffle=False, **loader_kw)

    model, head = build_model(arch)
    model.to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats(device)
    t0, best_f1, best_epoch, best_state, wait, history = time.time(), -1.0, 0, None, 0, []
    # progress is saved after every epoch, so a dropped session loses at most one epoch
    resume_path, start, prev_seconds, stopped = OUT / "resume" / f"{name}.pt", 1, 0.0, False
    if resume_path.exists() and not smoke:
        r = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(r["model"]); opt.load_state_dict(r["opt"]); g.set_state(r["g"])
        torch.set_rng_state(r["torch_rng"]); np.random.set_state(r["np_rng"]); random.setstate(r["py_rng"])
        best_f1, best_epoch, best_state, wait = r["best_f1"], r["best_epoch"], r["best_state"], r["wait"]
        history, prev_seconds, stopped, start = r["history"], r["seconds"], r["stopped"], r["epoch"] + 1
        print(f"resume {name} after epoch {r['epoch']}")
    for epoch in range(start, (1 if smoke else MAX_EPOCHS) + 1):
        if stopped:
            break
        # train mode: only the head gets gradients, but BatchNorm running statistics
        # in the backbone still update
        model.train()
        for x, y in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x.to(device, non_blocking=True)), y.to(device, non_blocking=True))
            loss.backward(); opt.step()
        val = scores(*predict(model, val_loader, device))
        unique = len(set(sampler.last)) if oversample else len(train_df)
        history.append({"epoch": epoch, "unique_train_images": unique, **val})
        print(f"{name} epoch {epoch}: val acc {val['acc']:.2f} macro F1 {val['macro_f1']:.2f}")
        if val["macro_f1"] > best_f1:
            best_f1, best_epoch, wait = val["macro_f1"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
        stopped = wait >= PATIENCE
        if not smoke:
            resume_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "g": g.get_state(),
                        "torch_rng": torch.get_rng_state(), "np_rng": np.random.get_state(), "py_rng": random.getstate(),
                        "best_f1": best_f1, "best_epoch": best_epoch, "best_state": best_state, "wait": wait,
                        "history": history, "seconds": prev_seconds + time.time() - t0, "stopped": stopped},
                       str(resume_path) + ".tmp")
            os.replace(str(resume_path) + ".tmp", resume_path)  # atomic, so a disconnect mid-save can't corrupt it
        if stopped:
            break
    train_seconds = prev_seconds + time.time() - t0
    model.load_state_dict(best_state)

    results, all_logits = {}, {}
    conditions = [("clean", "none", None)] + TEST_CONDITIONS
    for family, severity, value in conditions:
        pre = None if family == "clean" else test_corruption(family, value)
        loader = DataLoader(ImageDataset(test_df, eval_tf, pre), shuffle=False, **loader_kw)
        logits, labels = predict(model, loader, device)
        key = family if family == "clean" else f"{family}_{severity}"
        results[key] = scores(logits, labels); all_logits[key] = logits.astype(np.float16)
    corr = [results[f"{f}_{s}"] for f, s, _ in TEST_CONDITIONS]

    out = {
        "run": name, "arch": arch, "variant": variant, "seed": seed, "oversample": oversample,
        "corrupt_before_resize": corrupt_before_resize, "smoke": smoke,
        "total_params": total, "trainable_params": trainable,
        "epochs_trained": len(history), "best_epoch": best_epoch, "history": history,
        "train_seconds": round(train_seconds, 1),
        "peak_gpu_mem_mb": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "results": results,
        "mean_corr_acc": float(np.mean([r["acc"] for r in corr])),
        "mean_corr_macro_f1": float(np.mean([r["macro_f1"] for r in corr])),
        "worst_acc": float(min(r["acc"] for r in corr)),
        "worst_macro_f1": float(min(r["macro_f1"] for r in corr)),
        "mean_unique_train_images": float(np.mean([h["unique_train_images"] for h in history])),
        "torch": torch.__version__,
    }
    for d in ("results", "logits", "checkpoints"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(logit_path, labels=labels, **all_logits)
    torch.save(best_state, OUT / "checkpoints" / f"{name}.pth")
    json.dump(out, open(res_path, "w"), indent=2)  # written last: its presence means the run is complete
    if resume_path.exists():
        resume_path.unlink()
    print(f"done {name}: clean {results['clean']['acc']:.2f} | mean corr {out['mean_corr_acc']:.2f} | "
          f"worst {out['worst_acc']:.2f} | {train_seconds/60:.1f} min | {out['peak_gpu_mem_mb']} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True,
                    choices=["resnet50", "densenet121", "efficientnet_b0", "efficientnet_v2_s", "mobilenet_v3_large"])
    ap.add_argument("--variant", default="A0", choices=list(VARIANTS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--oversample", action="store_true")
    ap.add_argument("--corrupt_before_resize", action="store_true",
                    help="apply training corruptions before resizing, like the test corruptions")
    ap.add_argument("--smoke", action="store_true", help="500 images, 1 epoch, to check the pipeline")
    a = ap.parse_args()
    if a.smoke:
        OUT = OUT / "smoke"
    device = torch.device("cuda")
    data = load_splits()
    for s in a.seeds:
        run(a.arch, a.variant, s, a.oversample, a.corrupt_before_resize, data, device, a.smoke)
