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

# Random Seeds for reproducibility
RANDOM_STATE = 42

# Model Architecture Constants - SIMPLIFIED ARCHITECTURE (Numerical + Title)
TITLE_EMB_DIM = 200         # GloVe title embeddings (200D)
NUMERICAL_DIM = 36          # Enhanced: 34 original + domain_mean + user_mean

# Neural Network Configuration - SINGLE CONFIG ONLY
LEARNING_RATE = 2e-3        # Learning rate
HIDDEN_DIM = 128            # Hidden layer size
DROPOUT_RATE = 0.1          # Dropout rate
WEIGHT_DECAY = 1e-5         # Weight decay for regularization
PATIENCE = 50               # Early stopping patience (should be > MAX_EPOCHS if you want to train all epochs)
MAX_EPOCHS = 10             # Maximum epochs to train

# Training Configuration
VALIDATION_FREQUENCY = 1    # Validate every N epochs (set to 1 for every epoch)
GRADIENT_CLIP_NORM = 1.0    # Gradient clipping max norm
SCHEDULER_FACTOR = 0.5      # LR scheduler reduction factor
SCHEDULER_PATIENCE = 50     # LR scheduler patience

# Performance Thresholds
TARGET_R2 = 0.2             # Target R² score for the model
CLOSE_PERFORMANCE_THRESHOLD = 0.003  # Threshold for "close" performance comparison

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