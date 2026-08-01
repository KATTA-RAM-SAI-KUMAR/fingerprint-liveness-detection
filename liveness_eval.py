import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
)

from utils.dataset import get_dataloaders
from utils.model import build_model

# ==========================================
# CONFIG
# ==========================================

DATASET_PATH = "dataset"

MODEL_PATH = "models/liveness_model.pth"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("Fingerprint PAD Evaluation")
print("=" * 60)

print("Device:", DEVICE)

# ==========================================
# DATASET
# ==========================================

_, val_loader, test_loader = get_dataloaders(DATASET_PATH)

print("\nClass Mapping")

print(test_loader.dataset.class_to_idx)

# ==========================================
# MODEL
# ==========================================

model = build_model()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.to(DEVICE)

model.eval()

print("\nModel Loaded.")

# ==========================================
# INFERENCE
# ==========================================

scores = []

labels = []

predictions = []

times = []

with torch.no_grad():

    for images, target in test_loader:

        images = images.to(DEVICE)

        start = time.perf_counter()

        outputs = model(images)

        end = time.perf_counter()

        probs = F.softmax(outputs, dim=1)

        live_scores = probs[:,0]

        pred = torch.argmax(
            outputs,
            dim=1
        )

        scores.extend(
            live_scores.cpu().numpy()
        )

        labels.extend(
            target.numpy()
        )

        predictions.extend(
            pred.cpu().numpy()
        )

        times.append(
            (end-start)/len(images)
        )

scores=np.array(scores)

labels=np.array(labels)

predictions=np.array(predictions)

print()

print("Test Images :",len(scores))

print(
    "Average Inference Time:",
    round(np.mean(times)*1000,2),
    "ms"
)
scores = np.array(scores)
labels = np.array(labels)
predictions = np.array(predictions)
# ==========================================
# THRESHOLD SWEEP
# ==========================================

thresholds = np.arange(0.0, 1.01, 0.01)

results = []

for threshold in thresholds:

    # LIVE if probability >= threshold
    pred = np.where(scores >= threshold, 0, 1)

    live_mask = labels == 0
    spoof_mask = labels == 1

    total_live = np.sum(live_mask)
    total_spoof = np.sum(spoof_mask)

    false_reject = np.sum(pred[live_mask] == 1)

    false_accept = np.sum(pred[spoof_mask] == 0)

    bpcer = (
        false_reject / total_live
        if total_live > 0 else 0
    )

    apcer = (
        false_accept / total_spoof
        if total_spoof > 0 else 0
    )

    results.append({
        "threshold": threshold,
        "apcer": apcer,
        "bpcer": bpcer
    })
# ==========================================
# SELECT OPERATING THRESHOLD
# ==========================================

TARGET_BPCER = 0.03

valid = [
    r for r in results
    if r["bpcer"] <= TARGET_BPCER
]

if len(valid) > 0:

    best = min(
        valid,
        key=lambda x: x["apcer"]
    )

else:

    best = min(
        results,
        key=lambda x: abs(
            x["bpcer"] - TARGET_BPCER
        )
    )

best_threshold = best["threshold"]

best_bpcer = best["bpcer"]

best_apcer = best["apcer"]
# ==========================================
# EER
# ==========================================

differences = [
    abs(
        r["apcer"] -
        r["bpcer"]
    )
    for r in results
]

idx = np.argmin(differences)

eer_threshold = results[idx]["threshold"]

eer = (
    results[idx]["apcer"] +
    results[idx]["bpcer"]
) / 2

# ==========================================
# PRINT RESULTS
# ==========================================

print()

print("=" * 60)

print("Threshold Calibration")

print("=" * 60)

print(f"Selected Threshold : {best_threshold:.2f}")

print(f"BPCER              : {best_bpcer:.4f}")

print(f"APCER              : {best_apcer:.4f}")

print(f"EER                : {eer:.4f}")

print(f"EER Threshold      : {eer_threshold:.2f}")

# ==========================================
# FINAL PREDICTIONS
# ==========================================

final_predictions = np.where(
    scores >= best_threshold,
    0,
    1
)

cm = confusion_matrix(
    labels,
    final_predictions
)

print()

print("=" * 60)
print("Confusion Matrix")
print("=" * 60)

print(cm)

print()

print("=" * 60)
print("Classification Report")
print("=" * 60)

print(
    classification_report(
        labels,
        final_predictions,
        target_names=["LIVE", "SPOOF"],
        digits=4
    )
)
# ==========================================
# CONFUSION MATRIX IMAGE
# ==========================================

plt.figure(figsize=(6,6))

plt.imshow(cm, interpolation="nearest", cmap="Blues")

plt.title("Confusion Matrix")

plt.colorbar()

classes = ["LIVE", "SPOOF"]

tick_marks = np.arange(len(classes))

plt.xticks(tick_marks, classes)

plt.yticks(tick_marks, classes)

plt.xlabel("Predicted")

plt.ylabel("Actual")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            color="red",
            fontsize=14
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )
)

plt.close()
# ==========================================
# SCORE DISTRIBUTION
# ==========================================

live_scores = scores[labels == 0]
spoof_scores = scores[labels == 1]

plt.figure(figsize=(10,6))

plt.hist(
    live_scores,
    bins=20,
    alpha=0.7,
    color="green",
    label="LIVE"
)

plt.hist(
    spoof_scores,
    bins=20,
    alpha=0.7,
    color="red",
    label="SPOOF"
)

plt.axvline(
    best_threshold,
    color="black",
    linestyle="--",
    linewidth=2,
    label=f"Threshold = {best_threshold:.2f}"
)

plt.title("Score Distribution")

plt.xlabel("LIVE Probability")

plt.ylabel("Frequency")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "score_distribution.png"
    )
)

plt.close()
# ==========================================
# APCER - BPCER CURVE
# ==========================================

apcer_values = [r["apcer"] for r in results]
bpcer_values = [r["bpcer"] for r in results]

plt.figure(figsize=(8,6))

plt.plot(
    bpcer_values,
    apcer_values,
    linewidth=2
)

plt.scatter(
    best_bpcer,
    best_apcer,
    color="red",
    s=80,
    label="Operating Threshold"
)

plt.xlabel("BPCER")

plt.ylabel("APCER")

plt.title("APCER vs BPCER Tradeoff")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "apcer_bpcer_curve.png"
    )
)

plt.close()
# ==========================================
# SAVE RESULTS
# ==========================================

with open(
    os.path.join(
        OUTPUT_DIR,
        "evaluation_results.txt"
    ),
    "w"
) as f:

    f.write("Fingerprint PAD Evaluation\n\n")

    f.write(f"Threshold : {best_threshold:.4f}\n")

    f.write(f"APCER : {best_apcer:.4f}\n")

    f.write(f"BPCER : {best_bpcer:.4f}\n")

    f.write(f"EER : {eer:.4f}\n")

    f.write(f"EER Threshold : {eer_threshold:.4f}\n")

    f.write(
        f"Average Inference Time : {np.mean(times)*1000:.2f} ms\n"
    )

print("\nEvaluation completed successfully.")

print("Outputs saved in:", OUTPUT_DIR)