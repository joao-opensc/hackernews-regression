"""
PyTorch model definition for the CombinedScorePredictor.
"""
import torch
import torch.nn as nn
import config as cfg

class CombinedScorePredictor(nn.Module):
    def __init__(self, n_domains, n_users, domain_emb_dim=16, user_emb_dim=24,
                 title_emb_dim=200, numerical_dim=36, hidden_dim=128, dropout=0.15):
        super(CombinedScorePredictor, self).__init__()

        # Learnable embeddings with MUCH smaller initialization
        self.domain_embedding = nn.Embedding(n_domains, domain_emb_dim)
        self.user_embedding = nn.Embedding(n_users, user_emb_dim)
        
        # FIXED: Much smaller initialization to prevent large negative R² at start
        nn.init.normal_(self.domain_embedding.weight, 0, 0.01)  # Reduced from 0.1 to 0.01
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)    # Reduced from 0.1 to 0.01

        # Calculate total input dimension (full architecture)
        total_input_dim = title_emb_dim + numerical_dim + domain_emb_dim + user_emb_dim
        print(f"🏗️  Model architecture: {title_emb_dim}D title + {numerical_dim}D numerical + {domain_emb_dim}D domain + {user_emb_dim}D user = {total_input_dim}D total")

        # Simpler neural network architecture for stable training
        self.model = nn.Sequential(
            # First layer
            nn.Linear(total_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            # Final layer
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Apply He initialization for ReLU layers
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization for ReLU activation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, title_emb, numerical_features, domain_ids, user_ids):
        # Get trainable embeddings for domains and users
        domain_emb = self.domain_embedding(domain_ids)
        user_emb = self.user_embedding(user_ids)

        # FIXED: Remove aggressive scaling that was causing instability
        # Instead use gentle scaling that preserves the small initialization
        
        # Title embeddings: gentle scaling to match numerical scale
        title_scaled = title_emb * 0.5  # Gentle scaling down from ~0.27 std
        
        # Domain/user embeddings: NO scaling to preserve small initialization
        # This prevents the huge negative R² at the start of training
        domain_scaled = domain_emb  # Keep as-is (std ~0.01)
        user_scaled = user_emb      # Keep as-is (std ~0.01)
        
        # Concatenate all features with numerical features dominating initially
        combined = torch.cat([
            title_scaled,           # Gently scaled title embeddings (200D, std ~0.14)
            numerical_features,     # Standardized numerical features (36D, std ~1.0) 
            domain_scaled,          # Small domain embeddings (16D, std ~0.01)
            user_scaled             # Small user embeddings (24D, std ~0.01)
        ], dim=1)                   # Total: 276D - numerical features dominate initially

        # Forward pass through the network
        return self.model(combined).squeeze(1)
    
    def get_embedding_regularization_loss(self):
        """Add L2 regularization for embeddings to prevent overfitting."""
        domain_reg = torch.norm(self.domain_embedding.weight, p=2)
        user_reg = torch.norm(self.user_embedding.weight, p=2)
        return 0.001 * (domain_reg + user_reg)  # Small regularization coefficient

def create_model(title_emb_dim=None, numerical_dim=None, hidden_dim=None, dropout=None):
    """Factory function to create model with config defaults."""
    
    # Use config defaults if not provided
    if title_emb_dim is None:
        title_emb_dim = cfg.TITLE_EMB_DIM
    if numerical_dim is None:
        numerical_dim = cfg.NUMERICAL_DIM
    if hidden_dim is None:
        hidden_dim = cfg.HIDDEN_DIM
    if dropout is None:
        dropout = cfg.DROPOUT_RATE
    
    return HackerNewsPredictor(
        title_emb_dim=title_emb_dim,
        numerical_dim=numerical_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

class HackerNewsPredictor(nn.Module):
    def __init__(self, title_emb_dim=None, numerical_dim=None, hidden_dim=None, dropout=None):
        super().__init__()
        
        # Use config defaults if not provided
        if title_emb_dim is None:
            title_emb_dim = cfg.TITLE_EMB_DIM
        if numerical_dim is None:
            numerical_dim = cfg.NUMERICAL_DIM
        if hidden_dim is None:
            hidden_dim = cfg.HIDDEN_DIM
        if dropout is None:
            dropout = cfg.DROPOUT_RATE
        
        # Domain/User embeddings (NOT USED in numerical_plus_title approach but kept for compatibility)
        self.domain_embedding = nn.Embedding(cfg.NUM_DOMAINS, hidden_dim // 4)
        self.user_embedding = nn.Embedding(cfg.NUM_USERS, hidden_dim // 4)
        
        # Initialize embeddings with smaller std
        nn.init.normal_(self.domain_embedding.weight, 0, 0.01)  # Reduced from 0.1 to 0.01
        nn.init.normal_(self.user_embedding.weight, 0, 0.01)    # Reduced from 0.1 to 0.01
        
        # Title processing layers
        self.title_fc1 = nn.Linear(title_emb_dim, hidden_dim)
        self.title_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Numerical features processing
        self.num_fc1 = nn.Linear(numerical_dim, hidden_dim)
        self.num_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Combined processing
        # Input: hidden_dim//2 (title) + hidden_dim//2 (numerical) + hidden_dim//4 (domain) + hidden_dim//4 (user)
        combined_dim = (hidden_dim // 2) + (hidden_dim // 2) + (hidden_dim // 4) + (hidden_dim // 4)
        # Simplified: combined_dim = hidden_dim + hidden_dim // 2 = 1.5 * hidden_dim
        
        self.combined_fc1 = nn.Linear(combined_dim, hidden_dim)
        self.combined_fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output = nn.Linear(hidden_dim // 2, 1)
        
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
        self.batch_norm3 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm4 = nn.BatchNorm1d(hidden_dim // 2)

    def forward(self, title_embeddings, numerical_features, domain_ids, user_ids):
        """
        Forward pass with ALL input types.
        
        Args:
            title_embeddings: (batch_size, title_emb_dim) - GloVe embeddings
            numerical_features: (batch_size, numerical_dim) - engineered features
            domain_ids: (batch_size,) - domain IDs
            user_ids: (batch_size,) - user IDs
        
        Note: In the current approach, domain/user embeddings are computed but not used.
        The model uses only numerical + title features.
        """
        
        # Process title embeddings with gentle scaling to reduce dominance
        title_scaled = title_embeddings * 0.1  # Scale down GloVe embeddings
        title_out = torch.relu(self.batch_norm1(self.title_fc1(title_scaled)))
        title_out = self.dropout(title_out)
        title_out = torch.relu(self.batch_norm2(self.title_fc2(title_out)))
        
        # Process numerical features (already standardized)
        num_out = torch.relu(self.batch_norm1(self.num_fc1(numerical_features)))
        num_out = self.dropout(num_out)
        num_out = torch.relu(self.batch_norm2(self.num_fc2(num_out)))
        
        # Domain and user embeddings (computed but may not be used in final prediction)
        domain_emb = self.domain_embedding(domain_ids)
        user_emb = self.user_embedding(user_ids)
        
        # Combine all features
        # Note: This creates a combined representation but in practice we might only use 
        # numerical + title features for the simplified approach
        combined = torch.cat([
            title_scaled,           # Gently scaled title embeddings (cfg.TITLE_EMB_DIM, std ~0.14)
            numerical_features,     # Standardized numerical features (cfg.NUMERICAL_DIM, std ~1.0)
            domain_emb,            # Domain embeddings (hidden_dim//4, std ~0.01)
            user_emb               # User embeddings (hidden_dim//4, std ~0.01)
        ], dim=1)
        
        # Final processing
        combined_out = torch.relu(self.batch_norm3(self.combined_fc1(combined)))
        combined_out = self.dropout(combined_out)
        combined_out = torch.relu(self.batch_norm4(self.combined_fc2(combined_out)))
        combined_out = self.dropout(combined_out)
        
        # Output prediction (log scale)
        return self.output(combined_out).squeeze()