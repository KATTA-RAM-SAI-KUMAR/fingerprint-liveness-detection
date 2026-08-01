import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.dataset import get_dataloaders
from utils.model import build_model
from utils.trainer import Trainer

from utils.plots import plot_training

# ==========================================
# Configuration
# ==========================================

DATASET_PATH = "dataset"

EPOCHS = 30

LEARNING_RATE = 0.0003

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 50)
print("Fingerprint Liveness Detection Training")
print("=" * 50)
print(f"Device : {DEVICE}")
print()

# ==========================================
# Dataset
# ==========================================

train_loader, val_loader, test_loader = get_dataloaders(DATASET_PATH)

print(f"Training batches   : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")
print(f"Test batches       : {len(test_loader)}")
print()

# ==========================================
# Model
# ==========================================

model = build_model()

model.to(DEVICE)

print("MobileNetV3-Small Loaded Successfully.")
print()

# ==========================================
# Loss Function
# ==========================================

criterion = nn.CrossEntropyLoss()

# ==========================================
# Optimizer
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

# ==========================================
# Learning Rate Scheduler
# ==========================================

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2,
)

# ==========================================
# Trainer
# ==========================================

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    device=DEVICE,
    epochs=EPOCHS,
    patience=5,
)

# ==========================================
# Train Model
# ==========================================

history = trainer.train()

plot_training(history)

print()
print("=" * 50)
print("Training Completed Successfully!")
print("=" * 50)
print("Best model saved at:")
print("models/checkpoints/best_model.pth")
print()
print("Final model saved at:")
print("models/liveness_model.pth")