import os
import random
import shutil

random.seed(42)

LIVE_DIR = "data/live"
SPOOF_REAL_DIR = "data/spoof"
SPOOF_GEN_DIR = "data/spoof_generated"

OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

LIVE_COUNT = 840


def make_dirs():

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "live"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "spoof"), exist_ok=True)


def split_files(files):

    random.shuffle(files)

    n = len(files)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    return (
        files[:train_end],
        files[train_end:val_end],
        files[val_end:]
    )


def copy_files(files, src_dir, dst_dir):

    for f in files:
        shutil.copy(
            os.path.join(src_dir, f),
            os.path.join(dst_dir, f)
        )


make_dirs()

# -----------------------------
# LIVE
# -----------------------------

live_files = [
    f for f in os.listdir(LIVE_DIR)
    if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))
]

live_files = random.sample(live_files, LIVE_COUNT)

live_train, live_val, live_test = split_files(live_files)

copy_files(live_train, LIVE_DIR, os.path.join(OUTPUT_DIR, "train", "live"))
copy_files(live_val, LIVE_DIR, os.path.join(OUTPUT_DIR, "val", "live"))
copy_files(live_test, LIVE_DIR, os.path.join(OUTPUT_DIR, "test", "live"))

# -----------------------------
# SPOOF
# -----------------------------

spoof_files = []

for f in os.listdir(SPOOF_REAL_DIR):
    if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
        spoof_files.append(("real", f))

for f in os.listdir(SPOOF_GEN_DIR):
    if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
        spoof_files.append(("generated", f))

random.shuffle(spoof_files)

train_end = int(len(spoof_files) * TRAIN_RATIO)
val_end = int(len(spoof_files) * (TRAIN_RATIO + VAL_RATIO))

train_spoof = spoof_files[:train_end]
val_spoof = spoof_files[train_end:val_end]
test_spoof = spoof_files[val_end:]


def copy_spoof(data, split):

    for folder, file in data:

        src = SPOOF_REAL_DIR if folder == "real" else SPOOF_GEN_DIR

        shutil.copy(
            os.path.join(src, file),
            os.path.join(OUTPUT_DIR, split, "spoof", file)
        )


copy_spoof(train_spoof, "train")
copy_spoof(val_spoof, "val")
copy_spoof(test_spoof, "test")

print("=" * 50)
print("Dataset Created Successfully")
print("=" * 50)

print()

print(f"Train LIVE   : {len(os.listdir('dataset/train/live'))}")
print(f"Train SPOOF  : {len(os.listdir('dataset/train/spoof'))}")

print()

print(f"Val LIVE     : {len(os.listdir('dataset/val/live'))}")
print(f"Val SPOOF    : {len(os.listdir('dataset/val/spoof'))}")

print()

print(f"Test LIVE    : {len(os.listdir('dataset/test/live'))}")
print(f"Test SPOOF   : {len(os.listdir('dataset/test/spoof'))}")