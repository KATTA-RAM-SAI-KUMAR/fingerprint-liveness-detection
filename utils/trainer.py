import copy
import os

import torch
from tqdm import tqdm


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        epochs,
        patience=5,
    ):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = epochs
        self.patience = patience

        # Early Stopping
        self.best_loss = float("inf")
        self.counter = 0

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def train(self):

        best_weights = copy.deepcopy(self.model.state_dict())

        os.makedirs("models/checkpoints", exist_ok=True)

        for epoch in range(self.epochs):

            print(f"\nEpoch {epoch + 1}/{self.epochs}")

            train_loss, train_acc = self.run_train()

            val_loss, val_acc = self.run_validation()

            # Update learning rate scheduler
            self.scheduler.step(val_loss)

            # Save history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)

            print(
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}%"
            )

            # Save best model based on validation loss
            if val_loss < self.best_loss:

                self.best_loss = val_loss

                self.counter = 0

                best_weights = copy.deepcopy(self.model.state_dict())

                torch.save(
                    best_weights,
                    "models/checkpoints/best_model.pth",
                )

                print("✅ Best model saved.")

            else:

                self.counter += 1

                print(f"No improvement ({self.counter}/{self.patience})")

            # Early stopping
            if self.counter >= self.patience:

                print("\n🛑 Early stopping triggered.")

                break

        # Load best model
        self.model.load_state_dict(best_weights)

        # Save final model
        torch.save(
            self.model.state_dict(),
            "models/liveness_model.pth",
        )

        print("\nTraining Finished Successfully!")

        return self.history

    def run_train(self):

        self.model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(self.train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    def run_validation(self):

        self.model.eval()

        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():

            for images, labels in self.val_loader:

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = outputs.max(1)

                total += labels.size(0)

                correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc