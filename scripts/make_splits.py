"""Build fixed train/val/test split CSVs for ZhangLab, RSNA and CheXpert (seed 42)."""
import os, json, argparse
import numpy as np
import pandas as pd

SEED = 42
ZHANG_ROOT = "/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray"
RSNA_ROOT = "/kaggle/input/competitions/rsna-pneumonia-detection-challenge"
CHEX_ROOT = "/kaggle/input/datasets/ashery/chexpert"
IMG_EXT = (".jpeg", ".jpg", ".png")

# CheXpert pathologies used for "abnormal" (No Finding and Support Devices excluded)
PATHOLOGIES = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
               "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
               "Pleural Effusion", "Pleural Other", "Fracture"]


def list_images(folder):
    return sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))


def zhanglab(rng):
    rows = []
    def add(paths, split, label):
        rows.extend({"path": p, "split": split, "label": label} for p in paths)

    tr_n = [f"train/NORMAL/{f}" for f in list_images(f"{ZHANG_ROOT}/train/NORMAL")]
    tr_p = [f"train/PNEUMONIA/{f}" for f in list_images(f"{ZHANG_ROOT}/train/PNEUMONIA")]
    tr_n = list(rng.permutation(tr_n))
    tr_p = list(rng.permutation(tr_p))

    add(tr_n[141:], "train", 0)
    add(tr_n[:141], "val", 0)
    add(tr_p[:141], "val", 1)
    add([f"test/NORMAL/{f}" for f in list_images(f"{ZHANG_ROOT}/test/NORMAL")], "test", 0)
    add([f"test/PNEUMONIA/{f}" for f in list_images(f"{ZHANG_ROOT}/test/PNEUMONIA")], "test", 1)
    return pd.DataFrame(rows)


def rsna(rng):
    info = pd.read_csv(f"{RSNA_ROOT}/stage_2_detailed_class_info.csv").drop_duplicates("patientId")
    labels = pd.read_csv(f"{RSNA_ROOT}/stage_2_train_labels.csv")

    # boxes per patient as JSON list of [x, y, w, h] in original 1024x1024 pixels
    pos = labels[labels["Target"] == 1]
    boxes = {pid: json.dumps(g[["x", "y", "width", "height"]].round(1).values.tolist())
             for pid, g in pos.groupby("patientId")}

    normal = list(rng.permutation(sorted(info.loc[info["class"] == "Normal", "patientId"])))
    opacity = list(rng.permutation(sorted(info.loc[info["class"] == "Lung Opacity", "patientId"])))

    rows = []
    def add(pids, split, label):
        for pid in pids:
            rows.append({"path": f"stage_2_train_images/{pid}.dcm", "split": split,
                         "label": label, "boxes": boxes.get(str(pid), "[]")})

    add(normal[:1000], "test", 0)
    add(opacity[:1000], "test", 1)
    add(normal[1000:1300], "val", 0)
    add(opacity[1000:1300], "val", 1)
    add(normal[1300:], "train", 0)
    return pd.DataFrame(rows)


def chexpert(rng):
    df = pd.read_csv(f"{CHEX_ROOT}/train.csv")
    df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    df["path"] = df["Path"].str.split("CheXpert-v1.0-small/").str[-1]
    df["patient"] = df["path"].str.extract(r"(patient\d+)")[0]

    is_normal = df["No Finding"] == 1
    is_abnormal = (~is_normal) & (df[PATHOLOGIES] == 1).any(axis=1)
    df["label"] = np.where(is_normal, 0, np.where(is_abnormal, 1, -1))
    df = df[df["label"] >= 0]   # drops uncertain-only / support-devices-only images

    groups = {p: g for p, g in df.groupby("patient")}
    patients = [str(p) for p in rng.permutation(sorted(groups))]

    def pick(n_each, split):
        nonlocal patients
        pool, n0, n1, k = [], 0, 0, 0
        while (n0 < n_each or n1 < n_each) and k < len(patients):
            g = groups[patients[k]]
            pool.append(g)
            n0 += int((g["label"] == 0).sum())
            n1 += int((g["label"] == 1).sum())
            k += 1
        patients = patients[k:]          # these patients are now used up
        pool = pd.concat(pool)
        sel = pd.concat([pool[pool["label"] == 0].sample(n_each, random_state=SEED),
                         pool[pool["label"] == 1].sample(n_each, random_state=SEED)])
        return sel.assign(split=split)

    test = pick(1000, "test")
    val = pick(300, "val")
    train = pd.concat([groups[p] for p in patients])
    train = train[train["label"] == 0].assign(split="train")
    return pd.concat([train, val, test])[["path", "split", "label", "patient"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/splits")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for name, fn in [("zhanglab", zhanglab), ("rsna", rsna), ("chexpert", chexpert)]:
        rng = np.random.default_rng(SEED)
        df = fn(rng)
        if "patient" in df.columns:
            assert (df.groupby("patient")["split"].nunique() == 1).all(), \
                f"{name}: a patient appears in more than one split"
        df.to_csv(f"{args.out}/{name}.csv", index=False)
        print(f"\n=== {name}: {len(df)} rows ===")
        print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()