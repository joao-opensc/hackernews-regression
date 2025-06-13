"""
Configuration file for the HackerNews Score Prediction project.
"""

# Data Constants
NUMBER_OF_SAMPLES = 4000000
MINIMUM_SCORE = 10
MAXIMUM_SCORE = 1000
MIN_TRESHOLD = 10000
MAX_AUGMENT_PER_BIN = 15000
TOTAL_BUDGET = 100000
NUM_DOMAINS = 200
NUM_USERS = 1000

# Model Architecture Constants - SIMPLIFIED ARCHITECTURE (Numerical + Title)
TITLE_EMB_DIM = 200         # GloVe title embeddings (200D)
NUMERICAL_DIM = 36          # Enhanced: 34 original + domain_mean + user_mean

# Neural Network Configuration Testing - Multiple configs will be tested
# The training script will automatically test these ranges:

# Single Config for Quick Testing (if needed)
LEARNING_RATE = 2e-3        # Good starting point for simplified architecture
HIDDEN_DIM = 128            # Balanced network size
DROPOUT_RATE = 0.1          # Moderate dropout
WEIGHT_DECAY = 1e-5         # Light regularization
PATIENCE = 300              # Patience for early stopping
MAX_EPOCHS = 10           # Maximum epochs per configuration test

# Data Split Constants
VAL_SIZE = 0.2              # 20% for test set
TEST_SIZE = 0.25            # 25% of remaining for validation (so 60% train, 20% val, 20% test)

# File Paths
DATA_PATH = "data/hackernews_full_data.parquet"
GLOVE_FILE = "data/glove.6B.200d.txt"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = f"{ARTIFACTS_DIR}/best_numerical_title_model.pth"
MODEL_CONFIG_PATH = f"{ARTIFACTS_DIR}/model_config.pkl"
SCALER_PATH = f"{ARTIFACTS_DIR}/scaler.pkl"

# Legacy paths (no longer used but kept for compatibility)
DOMAIN_ENCODER_PATH = f"{ARTIFACTS_DIR}/domain_encoder.pkl"
USER_ENCODER_PATH = f"{ARTIFACTS_DIR}/user_encoder.pkl" 