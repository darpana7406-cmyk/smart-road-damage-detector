"""
Remap RDD_SPLIT labels from 5-class → 4-class YOLO format.

Mapping:
    0 -> 0 (longitudinal crack)
    1 -> 1 (transverse crack)
    2 -> 2 (alligator crack)
    3 -> 2 (alligator crack - merged)
    4 -> 3 (pothole)

Output: D:\PBEL\data\yolo\{train,val,test}\{images,labels}
"""
import os
import shutil
from pathlib import Path
from tqdm import tqdm

SRC = r"D:\PBEL\data\raw\RDD_SPLIT"
DST = r"D:\PBEL\data\yolo"

REMAP = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3}


def remap_label(src_label, dst_label):
    with open(src_label, "r") as f:
        lines = f.readlines()
    out = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        old_cls = int(parts[0])
        if old_cls not in REMAP:
            continue
        new_cls = REMAP[old_cls]
        out.append(f"{new_cls} " + " ".join(parts[1:]))
    if out:
        with open(dst_label, "w") as f:
            f.write("\n".join(out))
        return True
    return False


def process_split(split):
    src_img_dir = os.path.join(SRC, split, "images")
    src_lbl_dir = os.path.join(SRC, split, "labels")
    dst_img_dir = os.path.join(DST, split, "images")
    dst_lbl_dir = os.path.join(DST, split, "labels")

    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_lbl_dir, exist_ok=True)

    imgs = [f for f in os.listdir(src_img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    kept = 0
    for img in tqdm(imgs, desc=f"{split}"):
        stem = Path(img).stem
        lbl_src = os.path.join(src_lbl_dir, stem + ".txt")
        if not os.path.exists(lbl_src):
            continue
        lbl_dst = os.path.join(dst_lbl_dir, stem + ".txt")
        if remap_label(lbl_src, lbl_dst):
            shutil.copy(os.path.join(src_img_dir, img), os.path.join(dst_img_dir, img))
            kept += 1
    print(f"  {split}: {kept} images kept")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        process_split(split)
    print("\nDone. Output:", DST)