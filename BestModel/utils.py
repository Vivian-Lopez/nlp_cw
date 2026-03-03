"""
Utility functions for data loading, metrics, and model management.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
import torch
import ast


def load_and_convert_labels(file_path, skiprows=4):
    """
    Load TSV file and convert labels to binary classification.
    
    Args:
        file_path: Path to TSV file
        skiprows: Number of rows to skip (disclaimer)
        
    Returns:
        pd.DataFrame with columns: par_id, text, label
    """
    df = pd.read_csv(file_path, sep='\t', skiprows=skiprows, header=None)
    df.columns = ['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label_original']
    
    # Convert to binary: 0,1 -> 0 (No PCL), 2,3,4 -> 1 (PCL)
    df['label'] = df['label_original'].apply(lambda x: 0 if x <= 1 else 1)
    
    return df[['par_id', 'text', 'label']]


def load_dev_data(file_path, training_data_path):
    """
    Load dev set from CSV format with par_ids and multi-label annotations.
    Maps par_ids to text from training data and aggregates annotations to binary.
    
    Args:
        file_path: Path to dev CSV file with par_ids and labels
        training_data_path: Path to training TSV for text mapping
        
    Returns:
        pd.DataFrame with columns: par_id, text, label
    """
    # Load training data to create par_id -> text mapping
    train_df = pd.read_csv(training_data_path, sep='\t', skiprows=4, header=None)
    train_df.columns = ['par_id', 'art_id', 'keyword', 'country_code', 'text', 'label_original']
    par_id_to_text = dict(zip(train_df['par_id'], train_df['text']))
    
    # Load dev data
    df = pd.read_csv(file_path)
    
    # Convert label from string representation of list to actual list
    df['label_array'] = df['label'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # Convert annotations to binary: take mean of annotations, threshold at 0.5
    # Arrays contain 7 annotators' labels; if majority says PCL (1), label as 1
    df['label'] = df['label_array'].apply(lambda arr: 1 if np.mean(arr) > 0.5 else 0)
    
    # Map par_ids to text
    df['text'] = df['par_id'].map(par_id_to_text)
    
    # Remove rows where text is missing (par_ids not in training set)
    df = df.dropna(subset=['text'])
    
    return df[['par_id', 'text', 'label']]


def load_test_data(file_path):
    """
    Load test set (no labels).
    
    Args:
        file_path: Path to test TSV file
        
    Returns:
        pd.DataFrame with columns: par_id, text
    """
    df = pd.read_csv(file_path, sep='\t', skiprows=4, header=None)
    df.columns = ['par_id', 'art_id', 'keyword', 'country_code', 'text']
    
    return df[['par_id', 'text']]


def compute_metrics(y_true, y_pred, y_probs=None, threshold=0.5):
    """
    Compute precision, recall, and F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels (0 or 1)
        y_probs: Predicted probabilities (for threshold tuning)
        threshold: Decision threshold for converting probabilities to labels
        
    Returns:
        dict: Metrics
    """
    if y_probs is not None:
        y_pred = (y_probs >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def compute_class_weights(labels):
    """
    Compute inverse frequency class weights.
    
    Args:
        labels: Class labels
        
    Returns:
        torch.Tensor: Class weights
    """
    unique_classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(unique_classes) * counts)
    weights = weights / weights.sum() * len(unique_classes)  # Normalize
    return torch.tensor(weights, dtype=torch.float32)


def find_best_threshold(y_true, y_probs, threshold_range=(0.1, 0.9, 0.01)):
    """
    Find optimal threshold that maximizes F1 score.
    
    Args:
        y_true: True labels
        y_probs: Predicted probabilities
        threshold_range: (min, max, step) for threshold search
        
    Returns:
        tuple: (best_threshold, best_f1, results_dict)
    """
    min_thresh, max_thresh, step = threshold_range
    thresholds = np.arange(min_thresh, max_thresh + step, step)
    
    best_f1 = -1
    best_threshold = 0.5
    results = {}
    
    for threshold in thresholds:
        metrics = compute_metrics(y_true, None, y_probs, threshold)
        f1 = metrics['f1']
        results[threshold] = metrics
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1, results


def save_predictions(predictions, output_path):
    """
    Save predictions to file.
    
    Args:
        predictions: List of 0/1 predictions
        output_path: Path to save file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for pred in predictions:
            f.write(f"{pred}\n")


def load_predictions(input_path):
    """
    Load predictions from file.
    
    Args:
        input_path: Path to prediction file
        
    Returns:
        list: Predictions
    """
    with open(input_path, 'r') as f:
        predictions = [int(line.strip()) for line in f]
    return predictions
