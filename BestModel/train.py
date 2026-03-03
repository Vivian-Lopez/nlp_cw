"""
Training script for DeBERTa-v3-base binary classification model.
"""

import os
import random
import numpy as np
import torch
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import DebertaV2TokenizerFast, DebertaV2ForSequenceClassification, get_linear_schedule_with_warmup
import config
from utils import (
    load_and_convert_labels, load_dev_data, compute_metrics,
    compute_class_weights, find_best_threshold, save_predictions
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PCLDataset(Dataset):
    """Custom dataset for PCL classification."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def train_epoch(model, train_loader, optimizer, scheduler, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.float()  # Ensure float32
        
        loss = criterion(logits, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, eval_loader, device):
    """Evaluate model and return probabilities."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.float()  # Ensure float32
            
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels)


def main():
    """Main training loop."""
    
    # Set seeds
    set_seed(config.SEED)
    
    # Verify data paths exist
    train_path = Path(config.TRAINING_DATA_PATH)
    dev_path = Path(config.DEV_DATA_PATH)
    
    assert train_path.exists(), f"Training data not found at {train_path}"
    assert dev_path.exists(), f"Dev data not found at {dev_path}"
    
    logger.info("=" * 70)
    logger.info("Starting DeBERTa-v3-base Training for PCL Binary Classification")
    logger.info("=" * 70)
    
    # Device setup
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("Loading training data...")
    train_df = load_and_convert_labels(config.TRAINING_DATA_PATH)
    logger.info(f"Training data: {len(train_df)} samples")
    logger.info(f"Label distribution: {train_df['label'].value_counts().to_dict()}")
    
    logger.info("Loading dev data...")
    dev_df = load_dev_data(config.DEV_DATA_PATH, config.TRAINING_DATA_PATH)
    logger.info(f"Dev data: {len(dev_df)} samples")
    logger.info(f"Label distribution: {dev_df['label'].value_counts().to_dict()}")
    
    # Create internal train/val split (90/10)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['text'].values,
        train_df['label'].values,
        test_size=1 - config.TRAIN_VAL_SPLIT,
        random_state=config.RANDOM_STATE,
        stratify=train_df['label'].values
    )
    
    logger.info(f"Internal split - Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Initialize tokenizer and model
    logger.info(f"Loading {config.MODEL_NAME}...")
    tokenizer = DebertaV2TokenizerFast.from_pretrained(config.TOKENIZER_NAME)
    model = DebertaV2ForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=2
    )
    model.to(device)
    
    # Create datasets
    train_dataset = PCLDataset(train_texts, train_labels, tokenizer, config.MAX_LENGTH)
    val_dataset = PCLDataset(val_texts, val_labels, tokenizer, config.MAX_LENGTH)
    dev_dataset = PCLDataset(
        dev_df['text'].values, dev_df['label'].values, tokenizer, config.MAX_LENGTH
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )
    
    # Compute class weights
    class_weights = compute_class_weights(train_labels)
    class_weights = class_weights.to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    
    # Loss function and optimizer
    criterion = CrossEntropyLoss(weight=class_weights, reduction='mean')
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.EPOCHS
    warmup_steps = int(total_steps * config.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING PHASE")
    logger.info("=" * 70)
    
    best_dev_f1 = -1
    patience_counter = 0
    best_model_path = Path("best_model.pt")
    
    for epoch in range(config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device)
        
        # Evaluate on validation set
        val_probs, val_labels = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_labels, None, val_probs, threshold=0.5)
        
        # Evaluate on dev set
        dev_probs, dev_labels = evaluate(model, dev_loader, device)
        dev_metrics = compute_metrics(dev_labels, None, dev_probs, threshold=0.5)
        
        logger.info(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev P: {dev_metrics['precision']:.4f} | "
            f"Dev R: {dev_metrics['recall']:.4f} | "
            f"Dev F1: {dev_metrics['f1']:.4f}"
        )
        
        # Early stopping based on dev F1
        if dev_metrics['f1'] > best_dev_f1:
            best_dev_f1 = dev_metrics['f1']
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"✓ Best model saved with F1={best_dev_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Load best model
    logger.info("\n" + "=" * 70)
    logger.info("THRESHOLD TUNING PHASE")
    logger.info("=" * 70)
    
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    
    # Find best threshold on dev set
    best_threshold, best_f1, threshold_results = find_best_threshold(
        dev_labels, dev_probs, 
        threshold_range=(config.THRESHOLD_MIN, config.THRESHOLD_MAX, config.THRESHOLD_STEP)
    )
    
    logger.info(f"Best threshold: {best_threshold:.2f}")
    logger.info(f"Best F1 at threshold: {best_f1:.4f}")
    
    # Final evaluation with best threshold
    final_metrics = compute_metrics(dev_labels, None, dev_probs, threshold=best_threshold)
    logger.info(f"Final Dev Metrics (threshold={best_threshold:.2f}):")
    logger.info(f"  Precision: {final_metrics['precision']:.4f}")
    logger.info(f"  Recall: {final_metrics['recall']:.4f}")
    logger.info(f"  F1: {final_metrics['f1']:.4f}")
    
    # Save best threshold for inference
    with open("best_threshold.txt", "w") as f:
        f.write(str(best_threshold))
    
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Best model saved to: {best_model_path.absolute()}")
    logger.info(f"Best threshold: {best_threshold:.2f}")
    logger.info(f"Best dev F1: {best_f1:.4f}")
    logger.info(f"Training completed successfully!")
    logger.info("=" * 70 + "\n")
    
    return {
        'model': model,
        'tokenizer': tokenizer,
        'best_threshold': best_threshold,
        'dev_metrics': final_metrics,
        'device': device
    }


if __name__ == "__main__":
    main()
