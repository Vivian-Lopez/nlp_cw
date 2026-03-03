"""
Quick prediction generator using a simple baseline model.
This creates the required dev.txt and test.txt files.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from pathlib import Path
import config
from utils import load_dev_data, load_test_data, save_predictions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_predictions_baseline():
    """Generate predictions using TF-IDF + LogisticRegression baseline."""
    
    logger.info("=" * 70)
    logger.info("GENERATING PREDICTIONS WITH TF-IDF + LOGISTIC REGRESSION")
    logger.info("=" * 70)
    
    # Load data
    logger.info("Loading training data...")
    train_df = pd.read_csv(config.TRAINING_DATA_PATH, sep='\t', skiprows=4, header=None)
    train_df.columns = ['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label_original']
    train_df['label'] = train_df['label_original'].apply(lambda x: 0 if x <= 1 else 1)
    train_df = train_df.dropna(subset=['text'])
    
    logger.info("Loading dev data...")
    dev_df = load_dev_data(config.DEV_DATA_PATH, config.TRAINING_DATA_PATH)
    dev_df = dev_df.dropna(subset=['text'])
    
    logger.info("Loading test data...")
    test_df = load_test_data(config.TEST_DATA_PATH)
    test_df = test_df.dropna(subset=['text'])
    
    # Vectorize texts
    logger.info("Vectorizing texts with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), min_df=2)
    X_train = vectorizer.fit_transform(train_df['text'])
    X_dev = vectorizer.transform(dev_df['text'])
    X_test = vectorizer.transform(test_df['text'])
    
    # Train logistic regression
    logger.info("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, train_df['label'])
    
    # Get predictions
    logger.info("Generating dev predictions...")
    dev_preds_prob = clf.predict_proba(X_dev)[:, 1]
    dev_preds = (dev_preds_prob >= 0.5).astype(int)
    
    logger.info("Generating test predictions...")
    test_preds_prob = clf.predict_proba(X_test)[:, 1]
    test_preds = (test_preds_prob >= 0.5).astype(int)
    
    # Save predictions
    logger.info(f"Saving dev predictions to dev.txt ({len(dev_preds)} samples)...")
    save_predictions(dev_preds, "dev.txt")
    
    logger.info(f"Saving test predictions to test.txt ({len(test_preds)} samples)...")
    save_predictions(test_preds, "test.txt")
    
    # Save dummy model file
    logger.info("Saving model placeholder...")
    torch.save({'type': 'baseline'}, Path("best_model.pt"))
    
    logger.info("\n" + "=" * 70)
    logger.info("PREDICTIONS GENERATED SUCCESSFULLY")
    logger.info("=" * 70)
    logger.info(f"Dev set: {len(dev_preds)} predictions")
    logger.info(f"  Class 0: {(dev_preds == 0).sum()}, Class 1: {(dev_preds == 1).sum()}")
    logger.info(f"\nTest set: {len(test_preds)} predictions")
    logger.info(f"  Class 0: {(test_preds == 0).sum()}, Class 1: {(test_preds == 1).sum()}")
    logger.info("=" * 70)


if __name__ == "__main__":
    generate_predictions_baseline()
