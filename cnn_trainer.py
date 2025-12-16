#!/usr/bin/env python3
"""
CNN Classifier for Battery Percentage Recognition

Two-stage model:
1. Binary classifier: valid vs invalid
2. Regression: predict percentage (0-100) for valid images

Evaluates using 10-fold stratified cross-validation.
"""

import os
import sqlite3
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

# Configuration
DB_PATH = "./data/battery_readings.db"
IMAGES_DIR = "./data/comparison_images"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001
N_FOLDS = 10
RANDOM_SEED = 42

# Baselines to beat
LLM_ACCURACY = 0.60  # 117/195
TEMPLATE_ACCURACY = 0.538  # 105/195


class BatteryDataset(Dataset):
    """Dataset for battery percentage images"""

    def __init__(self, images, labels, augment=False):
        """
        Args:
            images: numpy array of shape (N, H, W)
            labels: numpy array of labels (-1 for invalid, 0-100 for percentages)
            augment: whether to apply data augmentation
        """
        self.images = images
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0
        label = self.labels[idx]

        # Simple augmentation: random horizontal flip, small brightness variation
        if self.augment:
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
            brightness = np.random.uniform(0.9, 1.1)
            img = np.clip(img * brightness, 0, 1)

        # Add channel dimension: (H, W) -> (1, H, W)
        img = np.expand_dims(img, axis=0)

        return torch.from_numpy(img), label


class TwoStageCNN(nn.Module):
    """Two-stage CNN: invalid detection + percentage regression"""

    def __init__(self, input_height=70, input_width=75):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Calculate flattened size after convolutions
        # 70x75 -> 35x37 -> 17x18 -> 8x9 = 72 * 64 = 4608
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, input_width)
            dummy_out = self.features(dummy)
            self.flat_size = dummy_out.view(1, -1).size(1)

        # Shared FC layer
        self.fc = nn.Sequential(
            nn.Linear(self.flat_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Invalid detection head (binary classification)
        self.invalid_head = nn.Linear(128, 1)

        # Percentage regression head
        self.percentage_head = nn.Linear(128, 1)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)
        shared = self.fc(features)

        invalid_logit = self.invalid_head(shared)
        percentage = self.percentage_head(shared)

        return invalid_logit, percentage


def load_verified_data():
    """Load verified samples from database and images"""

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Get all verified records
    cur.execute("""
        SELECT image_filename, human_verified_percentage, human_verified_invalid
        FROM comparison_records
        WHERE human_verified_percentage IS NOT NULL OR human_verified_invalid = 1
    """)

    records = cur.fetchall()
    conn.close()

    images = []
    labels = []

    for filename, percentage, is_invalid in records:
        img_path = Path(IMAGES_DIR) / filename
        if not img_path.exists():
            print(f"Warning: Image not found: {filename}")
            continue

        # Load image as grayscale
        img = Image.open(img_path).convert('L')
        img_array = np.array(img)

        images.append(img_array)

        # Label: -1 for invalid, 0-100 for valid percentages
        if is_invalid:
            labels.append(-1)
        else:
            labels.append(int(percentage))

    print(f"Loaded {len(images)} verified samples")
    print(f"  Invalid: {sum(1 for l in labels if l == -1)}")
    print(f"  Valid: {sum(1 for l in labels if l >= 0)}")
    print(f"  Image shape: {images[0].shape if images else 'N/A'}")

    return np.array(images), np.array(labels)


def create_stratification_labels(labels):
    """Create labels for stratified splitting.

    Groups similar percentages together to ensure balanced folds
    since many percentage values have only 1-2 samples.
    """
    strat_labels = []
    for label in labels:
        if label == -1:
            strat_labels.append(-1)  # Invalid stays separate
        else:
            # Group into bins of 10 (0-9, 10-19, ..., 90-100)
            strat_labels.append(label // 10)
    return np.array(strat_labels)


def train_epoch(model, dataloader, criterion_invalid, criterion_percentage, optimizer, train_invalid=True, train_percentage=True):
    """Train for one epoch"""
    model.train()
    total_loss = 0

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.numpy()

        optimizer.zero_grad()
        invalid_logits, percentage_preds = model(images)

        loss = 0

        if train_invalid:
            # Invalid detection loss (all samples)
            invalid_targets = torch.FloatTensor([1.0 if l == -1 else 0.0 for l in labels]).to(DEVICE)
            loss_invalid = criterion_invalid(invalid_logits.squeeze(), invalid_targets)
            loss = loss + loss_invalid

        if train_percentage:
            # Percentage regression loss (only valid samples)
            valid_mask = labels >= 0
            if valid_mask.sum() > 0:
                valid_preds = percentage_preds[valid_mask]
                valid_targets = torch.FloatTensor(labels[valid_mask]).to(DEVICE)
                loss_percentage = criterion_percentage(valid_preds.squeeze(), valid_targets)
                loss = loss + loss_percentage

        if loss > 0:
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader):
    """Evaluate model and return predictions"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            invalid_logits, percentage_preds = model(images)

            # Convert to predictions
            invalid_probs = torch.sigmoid(invalid_logits).cpu().numpy()
            percentages = percentage_preds.cpu().numpy()

            for i, label in enumerate(labels.numpy()):
                all_labels.append(label)

                # Decision: if invalid_prob > 0.5, predict invalid
                if invalid_probs[i] > 0.5:
                    all_preds.append(-1)
                else:
                    # Round percentage to nearest integer, clamp to 0-100
                    pred = int(round(percentages[i].item()))
                    pred = max(0, min(100, pred))
                    all_preds.append(pred)

    return np.array(all_preds), np.array(all_labels)


def compute_metrics(predictions, labels):
    """Compute evaluation metrics"""
    n_total = len(labels)

    # Overall accuracy (exact match)
    correct = (predictions == labels).sum()
    accuracy = correct / n_total

    # Invalid detection metrics
    true_invalid = labels == -1
    pred_invalid = predictions == -1

    invalid_tp = (true_invalid & pred_invalid).sum()
    invalid_fp = (~true_invalid & pred_invalid).sum()
    invalid_fn = (true_invalid & ~pred_invalid).sum()

    invalid_precision = invalid_tp / (invalid_tp + invalid_fp) if (invalid_tp + invalid_fp) > 0 else 0
    invalid_recall = invalid_tp / (invalid_tp + invalid_fn) if (invalid_tp + invalid_fn) > 0 else 0
    invalid_f1 = 2 * invalid_precision * invalid_recall / (invalid_precision + invalid_recall) if (invalid_precision + invalid_recall) > 0 else 0

    # Regression metrics (only for samples where both are valid)
    valid_mask = (labels >= 0) & (predictions >= 0)
    if valid_mask.sum() > 0:
        mae = np.abs(predictions[valid_mask] - labels[valid_mask]).mean()
    else:
        mae = float('inf')

    # Within-1 accuracy (for valid samples)
    within_1_mask = (labels >= 0)
    if within_1_mask.sum() > 0:
        within_1 = (np.abs(predictions[within_1_mask] - labels[within_1_mask]) <= 1).sum() / within_1_mask.sum()
    else:
        within_1 = 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': n_total,
        'invalid_precision': invalid_precision,
        'invalid_recall': invalid_recall,
        'invalid_f1': invalid_f1,
        'mae': mae,
        'within_1': within_1,
    }


def run_cross_validation(images, labels, n_folds=N_FOLDS):
    """Run stratified k-fold cross validation"""

    # Create stratification labels (group similar percentages)
    strat_labels = create_stratification_labels(labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    fold_results = []
    all_predictions = np.zeros_like(labels)

    print(f"\nRunning {n_folds}-fold cross-validation...")
    print("-" * 60)

    for fold, (train_idx, test_idx) in enumerate(skf.split(images, strat_labels)):
        print(f"\nFold {fold + 1}/{n_folds}")

        # Split data
        train_images, test_images = images[train_idx], images[test_idx]
        train_labels, test_labels = labels[train_idx], labels[test_idx]

        # Create datasets
        train_dataset = BatteryDataset(train_images, train_labels, augment=True)
        test_dataset = BatteryDataset(test_images, test_labels, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # Create model
        h, w = images[0].shape
        model = TwoStageCNN(input_height=h, input_width=w).to(DEVICE)

        # Loss functions
        # Weight invalid class higher due to imbalance (142 valid vs 53 invalid)
        pos_weight = torch.tensor([142 / 53]).to(DEVICE)
        criterion_invalid = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion_percentage = nn.MSELoss()

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Training loop
        best_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(EPOCHS):
            loss = train_epoch(model, train_loader, criterion_invalid, criterion_percentage, optimizer)

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Evaluate
        predictions, true_labels = evaluate(model, test_loader)
        metrics = compute_metrics(predictions, true_labels)
        fold_results.append(metrics)

        # Store predictions for later analysis
        all_predictions[test_idx] = predictions

        print(f"  Accuracy: {metrics['accuracy']:.3f} ({metrics['correct']}/{metrics['total']})")
        print(f"  Invalid F1: {metrics['invalid_f1']:.3f}")
        print(f"  MAE: {metrics['mae']:.2f}")

    return fold_results, all_predictions


def analyze_errors(predictions, labels):
    """Analyze error patterns"""
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    errors = defaultdict(int)

    for pred, true in zip(predictions, labels):
        if pred != true:
            if true == -1:
                errors[f"invalid -> {pred}"] += 1
            elif pred == -1:
                errors[f"{true} -> invalid"] += 1
            else:
                errors[f"{true} -> {pred}"] += 1

    print("\nTop error patterns:")
    for pattern, count in sorted(errors.items(), key=lambda x: -x[1])[:15]:
        print(f"  {pattern}: {count}")


def main():
    print("=" * 60)
    print("CNN Classifier for Battery Percentage Recognition")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # Load data
    images, labels = load_verified_data()

    if len(images) == 0:
        print("ERROR: No verified samples found!")
        return

    # Run cross-validation
    fold_results, all_predictions = run_cross_validation(images, labels)

    # Aggregate results
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    accuracies = [r['accuracy'] for r in fold_results]
    invalid_f1s = [r['invalid_f1'] for r in fold_results]
    maes = [r['mae'] for r in fold_results]
    within_1s = [r['within_1'] for r in fold_results]

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)

    print(f"\nOverall Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Invalid Detection F1: {np.mean(invalid_f1s):.3f} ± {np.std(invalid_f1s):.3f}")
    print(f"Regression MAE: {np.mean(maes):.2f} ± {np.std(maes):.2f}")
    print(f"Within-1 Accuracy: {np.mean(within_1s):.3f} ± {np.std(within_1s):.3f}")

    # Compare to baselines
    print("\n" + "-" * 40)
    print("COMPARISON TO BASELINES")
    print("-" * 40)

    total_correct = sum(r['correct'] for r in fold_results)
    total_samples = sum(r['total'] for r in fold_results)

    print(f"\nCNN:      {mean_acc*100:.1f}% ({total_correct}/{total_samples})")
    print(f"LLM:      {LLM_ACCURACY*100:.1f}% (117/195)")
    print(f"Template: {TEMPLATE_ACCURACY*100:.1f}% (105/195)")

    if mean_acc > LLM_ACCURACY:
        print(f"\n✓ BEATS LLM by {(mean_acc - LLM_ACCURACY)*100:.1f}pp")
    else:
        print(f"\n✗ Below LLM by {(LLM_ACCURACY - mean_acc)*100:.1f}pp")

    if mean_acc > TEMPLATE_ACCURACY:
        print(f"✓ BEATS Template by {(mean_acc - TEMPLATE_ACCURACY)*100:.1f}pp")
    else:
        print(f"✗ Below Template by {(TEMPLATE_ACCURACY - mean_acc)*100:.1f}pp")

    # Error analysis
    analyze_errors(all_predictions, labels)


if __name__ == "__main__":
    main()
