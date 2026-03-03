"""
Inference script for generating predictions on dev and test sets.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import DebertaV2TokenizerFast, DebertaV2ForSequenceClassification
import config
from utils import load_dev_data, load_test_data, save_predictions
from train import PCLDataset

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_trained_model(model_path, device):
    """Load trained model and tokenizer."""
    model = DebertaV2ForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=2
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    tokenizer = DebertaV2TokenizerFast.from_pretrained(config.TOKENIZER_NAME)
    
    return model, tokenizer


def predict(model, dataloader, device):
    """Get predictions from model."""
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of positive class
            all_probs.extend(probs.cpu().numpy())
    
    return np.array(all_probs)


def generate_predictions_dev(model_path, best_threshold, output_path="dev.txt"):
    """Generate predictions for dev set."""
    
    logger.info("=" * 70)
    logger.info("GENERATING DEV SET PREDICTIONS")
    logger.info("=" * 70)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    model, tokenizer = load_trained_model(model_path, device)
    
    # Load dev data
    logger.info("Loading dev data...")
    dev_df = load_dev_data(config.DEV_DATA_PATH, config.TRAINING_DATA_PATH)
    logger.info(f"Dev data: {len(dev_df)} samples")
    
    # Create dataset and loader
    dev_dataset = PCLDataset(
        dev_df['text'].values, dev_df['label'].values, tokenizer, config.MAX_LENGTH
    )
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Get predictions
    logger.info("Running inference...")
    dev_probs = predict(model, dev_loader, device)
    
    # Apply threshold
    dev_preds = (dev_probs >= best_threshold).astype(int)
    
    # Save predictions
    logger.info(f"Saving predictions to {output_path}...")
    save_predictions(dev_preds, output_path)
    
    logger.info(f"Dev predictions saved: {len(dev_preds)} samples")
    logger.info(f"Class distribution: {np.bincount(dev_preds) if len(np.bincount(dev_preds)) > 1 else 'All same class'}")
    logger.info("=" * 70 + "\n")


def generate_predictions_test(model_path, best_threshold, output_path="test.txt"):
    """Generate predictions for test set."""
    
    logger.info("=" * 70)
    logger.info("GENERATING TEST SET PREDICTIONS")
    logger.info("=" * 70)
    
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {model_path}...")
    model, tokenizer = load_trained_model(model_path, device)
    
    # Load test data
    logger.info("Loading test data...")
    test_df = load_test_data(config.TEST_DATA_PATH)
    logger.info(f"Test data: {len(test_df)} samples")
    
    # Create dummy labels for dataset compatibility (not used in prediction)
    dummy_labels = np.zeros(len(test_df), dtype=int)
    
    # Create dataset and loader
    test_dataset = PCLDataset(
        test_df['text'].values, dummy_labels, tokenizer, config.MAX_LENGTH
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Get predictions
    logger.info("Running inference...")
    test_probs = predict(model, test_loader, device)
    
    # Apply threshold
    test_preds = (test_probs >= best_threshold).astype(int)
    
    # Save predictions
    logger.info(f"Saving predictions to {output_path}...")
    save_predictions(test_preds, output_path)
    
    logger.info(f"Test predictions saved: {len(test_preds)} samples")
    logger.info(f"Class distribution: {np.bincount(test_preds) if len(np.bincount(test_preds)) > 1 else 'All same class'}")
    logger.info("=" * 70 + "\n")


def main():
    """Generate predictions for both dev and test sets."""
    
    model_path = Path("best_model.pt")
    threshold_path = Path("best_threshold.txt")
    
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}. Run train.py first.")
        return
    
    # Load best threshold from file
    if threshold_path.exists():
        with open(threshold_path, "r") as f:
            best_threshold = float(f.read().strip())
        logger.info(f"Loaded best threshold from file: {best_threshold:.2f}")
    else:
        best_threshold = 0.5
        logger.warning(f"Threshold file not found, using default: {best_threshold}")
    
    # Generate dev predictions
    generate_predictions_dev(model_path, best_threshold, output_path="dev.txt")
    
    # Generate test predictions
    generate_predictions_test(model_path, best_threshold, output_path="test.txt")
    
    logger.info("All prediction files generated successfully!")


if __name__ == "__main__":
    main()
