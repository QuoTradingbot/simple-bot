"""
Exit Model Training Script
===========================
Trains the neural network to predict optimal exit parameters from exit experiences.

Uses ALL 208 features from the JSON:
- 132 exit_params_used (what was actually used)
- 76 other features (market_state, outcome, root fields, etc.)

Predicts 132 exit_params (what should be used next time)
"""

import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    logging.warning(f"PyTorch not available: {e}")
    TORCH_AVAILABLE = False

from exit_feature_extraction import prepare_training_data

# Only import neural model if torch is available
if TORCH_AVAILABLE:
    from neural_exit_model import ExitParamsNet

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


if TORCH_AVAILABLE:
    class ExitExperienceDataset(Dataset):
        """PyTorch dataset for exit experiences"""
        
        def __init__(self, X, y):
            self.X = torch.FloatTensor(X)
            self.y = torch.FloatTensor(y)
        
        def __len__(self):
            return len(self.X)
        
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]
else:
    class ExitExperienceDataset:
        """Dummy class when PyTorch is not available"""
        pass


def train_exit_model(
    experience_file='data/local_experiences/exit_experiences_v2.json',
    model_save_path='models/exit_model.pth',
    epochs=100,
    batch_size=32,
    learning_rate=0.001,
    validation_split=0.2
):
    """
    Train the exit neural network on historical experiences.
    
    Args:
        experience_file: Path to JSON file with exit experiences
        model_save_path: Where to save the trained model
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        validation_split: Fraction of data to use for validation
    """
    logger.info("="*80)
    logger.info("EXIT MODEL TRAINING")
    logger.info("="*80)
    
    if not TORCH_AVAILABLE:
        logger.error("❌ PyTorch is not available in this environment!")
        logger.info("\nTo train the model, you need to:")
        logger.info("  1. Install PyTorch: pip install torch")
        logger.info("  2. Or run this on a machine with PyTorch installed")
        logger.info("  3. The feature extraction works fine - only training needs PyTorch")
        return False
    
    # Load experiences
    logger.info(f"Loading experiences from: {experience_file}")
    try:
        with open(experience_file, 'r') as f:
            data = json.load(f)
        experiences = data.get('experiences', [])
        logger.info(f"✅ Loaded {len(experiences):,} exit experiences")
    except Exception as e:
        logger.error(f"❌ Failed to load experiences: {e}")
        return False
    
    if len(experiences) < 100:
        logger.error(f"❌ Need at least 100 experiences to train, got {len(experiences)}")
        return False
    
    # Prepare training data (208 input features, 132 output params)
    logger.info("Extracting features from experiences...")
    X, y = prepare_training_data(experiences)
    
    logger.info(f"✅ Prepared data:")
    logger.info(f"   Input features (X): {X.shape} - 208 features per experience")
    logger.info(f"   Output params (y): {y.shape} - 132 exit parameters to predict")
    
    # Create dataset
    dataset = ExitExperienceDataset(X, y)
    
    # Split into train/validation
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"   Training samples: {train_size:,}")
    logger.info(f"   Validation samples: {val_size:,}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    logger.info("\nInitializing neural network...")
    model = ExitParamsNet(input_size=208, hidden_size=256)
    logger.info(f"✅ Model architecture: 208 inputs → 256 → 256 → 256 → 132 outputs")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    logger.info(f"\nStarting training for {epochs} epochs...")
    logger.info("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 20
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Log progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            os.makedirs(os.path.dirname(model_save_path) if os.path.dirname(model_save_path) else 'models', exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"   ✅ New best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info(f"\nEarly stopping at epoch {epoch+1} (no improvement for {early_stop_patience} epochs)")
                break
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"✅ Best validation loss: {best_val_loss:.6f}")
    logger.info(f"✅ Model saved to: {model_save_path}")
    logger.info(f"\nThe model has learned to predict all 132 exit parameters from 208 input features.")
    logger.info("Next backtest will use this trained model for adaptive exit decisions.")
    
    return True


def main():
    """Main entry point"""
    # Default paths (can be overridden via command line)
    experience_file = 'data/local_experiences/exit_experiences_v2.json'
    model_save_path = 'models/exit_model.pth'
    
    # Check if experience file exists
    if not os.path.exists(experience_file):
        logger.error(f"❌ Experience file not found: {experience_file}")
        logger.info("Run a backtest first to generate exit experiences.")
        sys.exit(1)
    
    # Train the model
    success = train_exit_model(
        experience_file=experience_file,
        model_save_path=model_save_path,
        epochs=100,
        batch_size=32,
        learning_rate=0.001,
        validation_split=0.2
    )
    
    if not success:
        logger.error("❌ Training failed!")
        sys.exit(1)
    
    logger.info("\n✅ Training succeeded!")
    logger.info(f"\nTo use the trained model in your next backtest:")
    logger.info(f"  1. The model file is at: {model_save_path}")
    logger.info(f"  2. Run a backtest - it will automatically load and use the trained model")
    logger.info(f"  3. The bot will adapt exit parameters based on market conditions")


if __name__ == '__main__':
    main()
