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
LEARNING_RATE = 1e-3        # Learning rate (reduced from 2e-3 to avoid overshooting)
HIDDEN_DIM = 128            # Hidden layer size
DROPOUT_RATE = 0.15         # Dropout rate (increased from 0.1 for better regularization)
WEIGHT_DECAY = 5e-5         # Weight decay for regularization (increased from 1e-5)
PATIENCE = 25               # Early stopping patience (reduced from 50 to be reasonable vs MAX_EPOCHS)
MAX_EPOCHS = 100            # Maximum epochs to train (increased from 10!)

# Training Configuration
VALIDATION_FREQUENCY = 5       # Validate every N epochs (increased from 1 for speed)
GRADIENT_CLIP_NORM = 1.0       # Gradient clipping max norm
SCHEDULER_FACTOR = 0.7         # LR scheduler reduction factor (increased from 0.5 for gentler reduction)
SCHEDULER_PATIENCE = 15        # LR scheduler patience (reduced from 50 to be effective)

# Performance Monitoring Configuration (NEW - for speed optimization)
PRINT_FREQUENCY = 5            # Print progress every N epochs (instead of every epoch)
WANDB_LOG_FREQUENCY = 5        # Log to wandb every N epochs (instead of every epoch)
COMPUTE_R2_FREQUENCY = 5       # Compute R² every N epochs (expensive operation)
VERBOSE_TRAINING = False       # Set to True for detailed epoch-by-epoch output

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