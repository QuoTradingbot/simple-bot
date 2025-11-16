"""
Train Neural Network on Historical Experiences
Run this to create/update the trained model
"""
import sys
import os
# Add src to path so we import the updated model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from sklearn.model_selection import train_test_split

# Force reimport to get latest changes
import importlib
if 'neural_confidence_model' in sys.modules:
    importlib.reload(sys.modules['neural_confidence_model'])

from neural_confidence_model import SignalConfidenceNet

class TradingDataset(Dataset):
    """Dataset of historical trades for training"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def calculate_shaped_reward(exp, all_experiences):
    """
    BINARY CLASSIFICATION: Predict win (1.0) or loss (0.0).
    This is the simplest learnable target - just predict if trade will be profitable.
    
    Returns:
        float: 1.0 for wins, 0.0 for losses
    """
    pnl = exp.get('pnl', 0)
    return 1.0 if pnl > 0 else 0.0
    
def calculate_shaped_reward_OLD_COMPLEX(exp, all_experiences):
    """
    OLD COMPLEX VERSION - keeping for reference but not used.
    This was too complex and model couldn't learn from it.
    """
    pnl = exp.get('pnl', 0)
    atr = exp.get('atr', 2.0)
    
    # === PHASE 1.1: RISK-ADJUSTED BASE REWARD ===
    # Normalize P&L by risk taken (profit per unit of ATR)
    risk_per_contract = atr * 4.0  # Typical 4x ATR stop distance
    base_reward = pnl / risk_per_contract if risk_per_contract > 0 else 0
    # Range: Winning 2R trade = +2.0, losing 1R trade = -1.0
    
    # === PHASE 1.2: CAUSAL PENALTIES (Why trade was bad) ===
    penalty = 0.0
    
    # 1. Late Entry Penalty (poor timing = high slippage)
    slippage = exp.get('entry_slippage_ticks', 0)
    if slippage > 2.0:
        penalty += 0.5 * (slippage - 2.0)
    
    # 2. Volatility Risk Penalty (trading during vol spike)
    volatility_trend = exp.get('volatility_trend', 0)
    if volatility_trend > 0.3:  # Volatility rising >30%
        penalty += 1.0
    
    # 3. Wide Spread Penalty (low liquidity trap)
    spread = exp.get('bid_ask_spread_ticks', 0.5)
    if spread > 2.0:
        penalty += 0.5 * (spread - 2.0)
    
    # 4. Regime Shift Penalty (entered during dangerous conditions)
    regime = exp.get('market_regime', 'NORMAL')
    if regime in ['HIGH_VOL_CHOPPY', 'UNKNOWN']:
        penalty += 1.5
    
    # 5. Drawdown Risk Penalty (trading while underwater)
    drawdown_pct = abs(exp.get('drawdown_pct_at_entry', 0))
    if drawdown_pct > 10:
        penalty += 0.5 * (drawdown_pct - 10) / 10
    
    # 6. Fatigue Penalty (overtrading = revenge trading)
    time_since_last = exp.get('time_since_last_trade_mins', 999)
    if time_since_last < 5:
        penalty += 1.0
    
    # 7. Session Risk Penalty (trading at edges = low liquidity)
    hour = exp.get('hour', 12)
    if hour >= 20 or hour <= 1:  # Late night / early morning UTC
        penalty += 0.5
    
    # === PHASE 1.3: CAUSAL BONUSES (Why trade was good) ===
    bonus = 0.0
    
    # 1. High Confidence Bonus (model was certain and correct)
    confidence = exp.get('confidence', 0.5)
    if confidence > 0.8 and pnl > 0:
        bonus += 1.0 * (confidence - 0.8) / 0.2
    
    # 2. Good Timing Bonus (tight execution)
    if slippage < 0.5 and pnl > 0:
        bonus += 0.5
    
    # 3. Trend Following Bonus (traded with regime)
    if regime in ['NORMAL_TRENDING', 'HIGH_VOL_TRENDING'] and pnl > 0:
        bonus += 0.5
    
    # 4. Patience Bonus (waited for quality setup)
    if time_since_last > 30 and pnl > 0:
        bonus += 0.5
    
    # === PHASE 1.4: GHOST TRADE HANDLING ===
    # Penalize missing good trades, reward avoiding bad trades
    if not exp.get('took_trade', True):
        if pnl > 0:  # Ghost trade that would have won
            return -2.0  # Strong penalty for missing opportunity
        else:  # Ghost trade that would have lost
            return 1.0   # Reward for correctly avoiding bad setup
    
    # === PHASE 2.1: EXPLORATION BONUS ===
    # Reward discovering new patterns at low confidence
    if confidence < 0.6 and pnl > 0:
        bonus += 1.0 * (0.6 - confidence)  # More reward for lower confidence wins
    
    # === PHASE 2.2: CONSISTENCY COMPONENT ===
    # Penalize erratic performance (big wins/losses are risky)
    if abs(pnl) > (atr * 8):  # Outlier trade (>8 ATR move)
        penalty += 0.5  # Prefer consistent edges over lottery tickets
    
    # === PHASE 2.3: DRAWDOWN AVOIDANCE ===
    # Extra penalty for losing while already down
    consecutive_losses = exp.get('consecutive_losses', 0)
    if consecutive_losses > 2 and pnl < 0:
        penalty += 1.0 * (consecutive_losses - 2) / 3  # Escalating penalty
    
    # === PHASE 2.4: MULTI-OBJECTIVE FINAL REWARD ===
    profit_component = base_reward - penalty + bonus
    
    # Consistency component: Reward smaller, more reliable edges
    consistency_component = 0.0
    if 0 < pnl < (atr * 3):  # Small consistent win
        consistency_component = 0.5
    
    # Drawdown avoidance component: Extra reward for recovering
    drawdown_recovery_component = 0.0
    if consecutive_losses > 0 and pnl > 0:  # Breaking losing streak
        drawdown_recovery_component = 0.5 * min(consecutive_losses, 3)
    
    # Weighted combination (tunable)
    shaped_reward = (
        0.60 * profit_component +           # Primary: profit quality
        0.15 * consistency_component +      # Secondary: consistency
        0.15 * drawdown_recovery_component + # Secondary: recovery
        0.10 * (bonus - penalty * 0.5)      # Tertiary: execution quality
    )
    
    # Clip to reasonable range
    shaped_reward = max(-10.0, min(10.0, shaped_reward))
    
    return shaped_reward


def load_experiences():
    """Load and prepare training data from experiences"""
    print("=" * 80)
    print("LOADING TRAINING DATA")
    print("=" * 80)
    
    # Use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'local_experiences', 'signal_experiences_v2.json')
    
    with open(data_path) as f:
        data = json.load(f)
    
    experiences = data['experiences']
    
    # Include ALL experiences (both taken trades AND ghost trades)
    # Ghost trades teach the AI what it SHOULD have done
    print(f"Total experiences: {len(experiences):,}")
    taken_trades = [e for e in experiences if e.get('took_trade', True)]  # Include ghosts
    ghost_trades = [e for e in experiences if not e.get('took_trade', True)]
    print(f"Taken trades: {len(taken_trades):,}")
    print(f"Ghost trades (learning from rejections): {len(ghost_trades):,}")
    
    # Use ALL experiences for training (ghosts + taken)
    training_experiences = experiences  # Train on everything!

    
    # Prepare features and labels
    features = []
    labels = []
    
    # Session mapping
    session_map = {'Asia': 0, 'London': 1, 'NY': 2}
    signal_map = {'LONG': 0, 'SHORT': 1}
    trade_type_map = {'reversal': 0, 'continuation': 1}
    regime_map = {
        'NORMAL': 0,
        'NORMAL_TRENDING': 1,
        'HIGH_VOL_TRENDING': 2,
        'HIGH_VOL_CHOPPY': 3,
        'LOW_VOL_TRENDING': 4,
        'LOW_VOL_RANGING': 5,
        'UNKNOWN': 0  # Map unknown to NORMAL
    }
    
    for exp in training_experiences:
        # Extract 31 features (must match neural_confidence_model.py order!)
        # Use pre-calculated features from JSON when available, fallback to calculation for old data
        
        # Get pre-calculated features (or calculate as fallback for old experiences)
        minute = exp.get('minute')
        time_to_close = exp.get('time_to_close')
        price_mod_50 = exp.get('price_mod_50')
        
        # Fallback calculation if features not in JSON (for backward compatibility with old data)
        if minute is None or time_to_close is None or price_mod_50 is None:
            from datetime import datetime
            timestamp_str = exp.get('timestamp', '')
            if minute is None:
                minute = 0
                if timestamp_str:
                    try:
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        minute = dt.minute
                    except:
                        pass
            
            if time_to_close is None:
                time_to_close = 240  # default 4 hours
                if timestamp_str:
                    try:
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        hour_decimal = dt.hour + dt.minute / 60.0
                        time_to_close = max(0, 16.0 - hour_decimal) * 60
                    except:
                        pass
            
            if price_mod_50 is None:
                price = exp.get('price', exp.get('entry_price', 6500.0))
                price_mod_50 = (price % 50) / 50.0
        
        # Get price for normalization
        price = exp.get('price', exp.get('entry_price', 6500.0))
        
        feature_vec = [
            exp.get('rsi', 50.0),
            exp.get('vix', 15.0),
            exp.get('hour', 12),
            exp.get('atr', 2.0),
            exp.get('volume_ratio', 1.0),
            exp.get('vwap_distance', 0.0),
            exp.get('streak', 0),
            exp.get('consecutive_wins', 0),
            exp.get('consecutive_losses', 0),
            exp.get('cumulative_pnl_at_entry', 0.0),
            session_map.get(exp.get('session', 'NY'), 2),
            exp.get('trend_strength', 0.0),
            exp.get('sr_proximity_ticks', 0.0),
            trade_type_map.get(exp.get('trade_type', 'reversal'), 0),
            exp.get('time_since_last_trade_mins', 0.0),
            exp.get('drawdown_pct_at_entry', 0.0),
            exp.get('day_of_week', 0),
            exp.get('recent_pnl', 0.0),
            exp.get('commission_cost', 0.0),
            signal_map.get(exp.get('signal', 'LONG'), 0),
            # ADVANCED ML FEATURES
            regime_map.get(exp.get('market_regime', 'NORMAL'), 0),  # Market regime (0-4)
            exp.get('recent_volatility_20bar', 2.0),  # Rolling 20-bar price std
            exp.get('volatility_trend', 0.0),  # Is volatility increasing
            exp.get('vwap_std_dev', 2.0),  # VWAP standard deviation
            # NEW TEMPORAL/PRICE FEATURES (3 features)
            minute / 60.0,  # Minute of hour (0-1) - PRE-CALCULATED
            time_to_close / 240.0,  # Time to close normalized (0-1, 4hrs max) - PRE-CALCULATED
            price_mod_50,  # Distance to round 50-level (0-1) - PRE-CALCULATED
            # ADDITIONAL FEATURES (4 features - brings total to 31)
            exp.get('price_momentum', 0.0),  # Price momentum indicator
            exp.get('volume_momentum', 1.0),  # Volume momentum
            exp.get('spread_normalized', 0.0),  # Bid-ask spread normalized
            exp.get('liquidity_score', 1.0),  # Market liquidity score
        ]
        
        # BINARY SUCCESS: 1.0 if winning trade, 0.0 if losing trade
        pnl = exp.get('pnl', 0)
        label = 1.0 if pnl > 0 else 0.0
        
        features.append(feature_vec)
        labels.append(label)
    
    features = np.array(features, dtype=np.float32)
    labels = np.array(labels, dtype=np.float32)
    
    # Calculate win rate statistics
    win_rate = labels.mean() * 100
    total_trades = len(labels)
    winning_trades = labels.sum()
    print(f"Win Rate: {win_rate:.1f}% ({int(winning_trades)}/{total_trades} trades)")
    print()
    
    return features, labels, taken_trades


def normalize_features(features):
    """Normalize features to 0 mean, 1 std"""
    means = features.mean(axis=0)
    stds = features.std(axis=0)
    # If std is very small (< 0.01), replace with 1.0 to avoid extreme scaling
    stds = np.where(stds < 0.01, 1.0, stds)
    normalized = (features - means) / stds
    return normalized, means, stds


def train_model(epochs=150, batch_size=32, learning_rate=0.001):
    """Train the neural network"""
    print("=" * 80)
    print("TRAINING NEURAL NETWORK")
    print("=" * 80)
    print()
    
    # Load data
    features, labels, experiences = load_experiences()
    
    # Normalize
    features, feature_means, feature_stds = normalize_features(features)
    
    # Split train/validation 80/20 (no stratification for regression)
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print()
    
    # Create datasets
    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    print()
    
    model = SignalConfidenceNet()
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.BCELoss()  # Binary cross-entropy for success probability (0 or 1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_acc = 0.0
    patience = 20
    patience_counter = 0
    
    print("Starting training...")
    print()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_mae = 0.0  # Mean absolute error for rewards
        train_total = 0
        
        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device)
            labels_batch = labels_batch.to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(features_batch)
            
            # DEBUG: Check output range
            if torch.isnan(outputs).any() or (outputs < 0).any() or (outputs > 1).any():
                print(f"❌ Invalid outputs detected!")
                print(f"   Min: {outputs.min().item():.4f}, Max: {outputs.max().item():.4f}")
                print(f"   NaN count: {torch.isnan(outputs).sum().item()}")
                print(f"   Model output layer: {list(model.network.children())[-1]}")
                raise RuntimeError("Model outputs are outside [0,1] range - sigmoid not working!")
            
            loss = criterion(outputs, labels_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_mae += torch.abs(outputs - labels_batch).sum().item()
            train_total += labels_batch.size(0)
        
        train_mae = train_mae / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_total = 0
        
        with torch.no_grad():
            for features_batch, labels_batch in val_loader:
                features_batch = features_batch.to(device)
                labels_batch = labels_batch.to(device).unsqueeze(1)
                
                outputs = model(features_batch)
                loss = criterion(outputs, labels_batch)
                
                val_loss += loss.item()
                val_mae += torch.abs(outputs - labels_batch).sum().item()
                val_total += labels_batch.size(0)
        
        val_mae = val_mae / val_total
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss/len(train_loader):.4f} | Train MAE: {train_mae:.2f}")
            print(f"  Val Loss: {val_loss/len(val_loader):.4f} | Val MAE: {val_mae:.2f}")
            print()
        
        # Early stopping (lower MAE is better for regression)
        if val_mae < best_val_acc or best_val_acc == 0.0:  # Reusing variable for MAE
            best_val_acc = val_mae
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'neural_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'train_mae': train_mae,
                'val_mae': val_mae,
                'feature_means': feature_means,
                'feature_stds': feature_stds,
                'epoch': epoch,
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                print()
                break
    
    # Temperature scaling calibration on validation set
    print("=" * 80)
    print("CALIBRATING TEMPERATURE SCALING")
    print("=" * 80)
    
    model.eval()
    
    # Skip temperature calibration - use default
    best_temperature = 1.0
    print(f"Using default temperature: {best_temperature}")
    print("(Temperature calibration skipped)")
    
    if False:  # Disabled for now
        # Find optimal temperature using validation set
        best_ece = float('inf')  # Expected Calibration Error
        
        for temp in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            # Collect predictions with this temperature
            all_confidences = []
            all_labels = []
            
            with torch.no_grad():
                for features_batch, labels_batch in val_loader:
                    features_batch = features_batch.to(device)
                    labels_batch = labels_batch.to(device)
                    
                    # Get logits (before final sigmoid)
                    logits = model.network[:-1](features_batch)
                    
                    # Apply temperature
                    scaled_logits = logits / temp
                    confidences = torch.sigmoid(scaled_logits).squeeze()
                    
                    # Flatten and convert to numpy
                    conf_np = confidences.cpu().numpy().flatten()
                    label_np = labels_batch.cpu().numpy().flatten()
                    
                    all_confidences.extend(conf_np.tolist())
                    all_labels.extend(label_np.tolist())
            
            # Calculate Expected Calibration Error (ECE)
            all_confidences = np.array(all_confidences)
            all_labels = np.array(all_labels)
            
            # Bin predictions and calculate calibration error
            n_bins = 10
            ece = 0.0
            for i in range(n_bins):
                bin_lower = i / n_bins
                bin_upper = (i + 1) / n_bins
                
                in_bin = (all_confidences >= bin_lower) & (all_confidences < bin_upper)
                if in_bin.sum() > 0:
                    avg_confidence = all_confidences[in_bin].mean()
                    avg_accuracy = all_labels[in_bin].mean()
                    ece += abs(avg_confidence - avg_accuracy) * in_bin.sum()
            
            ece = ece / len(all_labels)
            
            print(f"Temperature {temp:.1f}: ECE = {ece:.4f}, Conf Range = [{all_confidences.min():.2f}, {all_confidences.max():.2f}]")
            
            if ece < best_ece:
                best_ece = ece
                best_temperature = temp
        
        print()
        print(f"[OK] Optimal temperature: {best_temperature:.1f} (ECE: {best_ece:.4f})")
        print()
    
    # Save model with optimized temperature
    model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'neural_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_mae': train_mae,
        'val_mae': best_val_acc,  # best_val_acc is actually MAE now
        'feature_means': feature_means,
        'feature_stds': feature_stds,
        'temperature': best_temperature,
        'epoch': epoch if 'epoch' in locals() else 0,
    }, model_path)
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: data/neural_model.pth")
    print()
    print("Next steps:")
    print("  1. Run full_backtest.py to test the neural network")
    print("  2. Compare performance to previous backtests")
    print("  3. Re-train periodically as more experiences are collected")
    print()


if __name__ == '__main__':
    train_model()
