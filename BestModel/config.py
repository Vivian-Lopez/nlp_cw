"""
Configuration for DeBERTa-v3-base binary classification model.
"""

# Model Configuration
MODEL_NAME = "microsoft/deberta-v3-base"
TOKENIZER_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 128

# Training Configuration
LEARNING_RATE = 2e-5
BATCH_SIZE = 8  # CPU optimized (balanced between speed and memory)
EPOCHS = 5  # Full training for better performance 
EARLY_STOPPING_PATIENCE = 2
WARMUP_RATIO = 0.1

# Optimization
OPTIMIZER = "AdamW"
SCHEDULER = "linear"

# Data Paths
TRAINING_DATA_PATH = "../NLPLabs-2024/Dont_Patronize_Me_Trainingset/dontpatronizeme_pcl.tsv"
DEV_DATA_PATH = "../dontpatronizeme/semeval-2022/practice splits/dev_semeval_parids-labels.csv"
TEST_DATA_PATH = "../dontpatronizeme/semeval-2022/TEST/task4_test.tsv"

# Split Configuration
TRAIN_VAL_SPLIT = 0.9  # 90% train, 10% val from training data
RANDOM_STATE = 42

# Threshold Tuning
THRESHOLD_MIN = 0.1
THRESHOLD_MAX = 0.9
THRESHOLD_STEP = 0.01

# Device
DEVICE = "cpu"  # CPU training (GPU has sm_6.1, PyTorch compiled for sm_7.0+)

# Reproducibility
SEED = 42
