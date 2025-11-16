"""
Train Exit Neural Network on Historical Exit Experiences
Learns optimal exit parameters based on market context
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from neural_exit_model import ExitParamsNet, normalize_exit_params

class ExitDataset(Dataset):
    """Dataset of historical exit experiences"""
    
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_exit_experiences():
    """Load exit experiences from JSON"""
    print("=" * 80)
    print("LOADING EXIT TRAINING DATA")
    print("=" * 80)
    
    # Use absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, '..', 'data', 'local_experiences', 'exit_experiences_v2.json')
    
    with open(data_path) as f:
        data = json.load(f)
    
    experiences = data['experiences']
    print(f"Total exit experiences: {len(experiences):,}")
    
    # Filter to winning trades only (learn from success)
    winning_exits = [e for e in experiences if e.get('win', False)]
    print(f"Winning exits (learning targets): {len(winning_exits):,}")
    print(f"Win rate: {len(winning_exits)/len(experiences)*100:.1f}%")
    print()
    
    # Prepare features and labels
    features = []
    labels = []
    
    # Regime mapping
    regime_map = {
        'NORMAL': 0,
        'NORMAL_TRENDING': 0,
        'OVERBOUGHT': 1,
        'OVERSOLD': 1,
        'CHOPPY': 2,
        'HIGH_VOLUME': 0,
        'LOW_VOLUME': 0,
        'TRENDING': 0,
        'UNKNOWN': 0
    }
    
    side_map = {'LONG': 0, 'long': 0, 'SHORT': 1, 'short': 1}
    
    for exp in experiences:
        try:
            # Extract fields (10 market_state + 63 outcome + 132 exit_params = 205 total)
            # 63 outcome = 57 numeric fields + 6 additional (exit_reason, side, 4 summaries)
            market_state = exp.get('market_state', {})
            outcome = exp.get('outcome', {})
            exit_params_current = exp.get('exit_params', {})
            
            # Build feature vector from ALL fields (normalized 0-1)
            # MARKET STATE (10 features)
            market_features = [
                market_state.get('rsi', 50.0) / 100.0,
                min(market_state.get('vix', 15.0) / 50.0, 1.0),
                min(market_state.get('atr', 2.0) / 10.0, 1.0),
                min(market_state.get('volume_ratio', 1.0) / 5.0, 1.0),
                market_state.get('hour', 12) / 24.0,
                market_state.get('day_of_week', 2) / 6.0,
                market_state.get('streak', 0) / 10.0,
                np.clip(market_state.get('recent_pnl', 0.0) / 1000.0, -1, 1),
                np.clip(market_state.get('vwap_distance', 0.0) / 50.0, -1, 1),
                np.clip(market_state.get('peak_pnl', 0.0) / 1000.0, 0, 1),
            ]
            
            # OUTCOME features - extract scalar numeric fields only (skip lists/dicts/strings)
            # Define explicit list of numeric outcome fields we want as features (sorted for consistency)
            numeric_outcome_fields = [
                'atr_change_percent', 'avg_atr_during_trade', 'bars_held', 'bars_until_breakeven',
                'bars_until_trailing', 'bid_ask_spread_ticks', 'breakeven_activated',
                'breakeven_activation_bar', 'commission_cost', 'contracts',
                'cumulative_pnl_before_trade', 'daily_loss_limit', 'daily_loss_proximity_pct',
                'daily_pnl_before_trade', 'day_of_week', 'drawdown_bars', 'duration',
                'duration_bars', 'entry_bar', 'entry_confidence', 'entry_hour', 'entry_minute',
                'entry_price', 'exit_bar', 'exit_hour', 'exit_minute',
                'exit_param_update_count', 'held_through_sessions', 'high_volatility_bars',
                'losses_in_last_5_trades', 'mae', 'max_drawdown_percent', 'max_profit_reached',
                'max_r_achieved', 'mfe', 'min_r_achieved', 'minutes_until_close',
                'opportunity_cost', 'peak_r_multiple', 'peak_unrealized_pnl', 'pnl',
                'profit_drawdown_from_peak', 'r_multiple', 'rejected_partial_count',
                'session', 'slippage_ticks', 'stop_adjustment_count', 'stop_hit',
                'time_in_breakeven_bars', 'trade_number_in_session', 'trailing_activated',
                'trailing_activation_bar', 'vix', 'volatility_regime_change',
                'volume_at_exit', 'win', 'wins_in_last_5_trades'
            ]
            
            # Handle exit_reason and side specially (categorical -> numeric)
            exit_reason = outcome.get('exit_reason', 'unknown')
            exit_map = {'stop_loss': 0, 'target': 0.25, 'time': 0.5, 'partial': 0.75, 'trailing': 1.0,
                       'profit_drawdown': 0.2, 'sideways_market_exit': 0.3, 'volatility_spike': 0.4,
                       'underwater_timeout': 0.35, 'stale_exit': 0.6, 'forced_flatten': 0.55}
            exit_reason_encoded = exit_map.get(exit_reason, 0)
            
            side = outcome.get('side', 'long')
            side_encoded = 0.0 if side.lower() == 'long' else 1.0
            
            # Extract summary stats from list fields
            decision_history = outcome.get('decision_history', [])
            decision_count = len(decision_history) if isinstance(decision_history, list) else 0
            
            exit_param_updates = outcome.get('exit_param_updates', [])
            update_count = len(exit_param_updates) if isinstance(exit_param_updates, list) else 0
            
            stop_adjustments = outcome.get('stop_adjustments', [])
            adjustment_count = len(stop_adjustments) if isinstance(stop_adjustments, list) else 0
            
            unrealized_pnl_history = outcome.get('unrealized_pnl_history', [])
            pnl_history_count = len(unrealized_pnl_history) if isinstance(unrealized_pnl_history, list) else 0
            
            # Build outcome feature vector with explicit fields only
            outcome_features = []
            for field in numeric_outcome_fields:
                val = outcome.get(field, 0.0)
                # Normalize based on field type
                if field in ['breakeven_activated', 'trailing_activated', 'stop_hit', 'held_through_sessions',
                            'volatility_regime_change', 'win']:
                    outcome_features.append(float(val))  # Boolean 0/1
                elif field in ['pnl', 'peak_unrealized_pnl', 'opportunity_cost', 'cumulative_pnl_before_trade',
                              'daily_pnl_before_trade']:
                    outcome_features.append(np.clip(val / 1000.0, -1, 1))  # +/- $1000
                elif field in ['r_multiple', 'max_r_achieved', 'min_r_achieved', 'peak_r_multiple']:
                    outcome_features.append(np.clip(val / 10.0, -1, 1))  # +/- 10R
                elif field in ['entry_hour', 'exit_hour']:
                    outcome_features.append(val / 24.0)  # 0-1
                elif field in ['entry_minute', 'exit_minute']:
                    outcome_features.append(val / 60.0)  # 0-1
                elif field in ['duration', 'bars_held', 'duration_bars', 'bars_until_breakeven',
                              'time_in_breakeven_bars', 'bars_until_trailing', 'entry_bar', 'exit_bar',
                              'breakeven_activation_bar', 'trailing_activation_bar', 'drawdown_bars',
                              'high_volatility_bars']:
                    outcome_features.append(min(val / 300.0, 1.0))  # Cap at 300 bars
                elif field in ['mae', 'mfe', 'max_profit_reached', 'profit_drawdown_from_peak']:
                    outcome_features.append(np.clip(val / 500.0, -1, 1))  # +/- $500
                elif field in ['exit_param_update_count', 'stop_adjustment_count', 'rejected_partial_count']:
                    outcome_features.append(min(val / 50.0, 1.0))  # Cap at 50
                elif field in ['atr_change_percent', 'max_drawdown_percent', 'daily_loss_proximity_pct']:
                    outcome_features.append(np.clip(val / 100.0, -1, 1))  # +/- 100%
                elif field in ['avg_atr_during_trade']:
                    outcome_features.append(min(val / 10.0, 1.0))  # Cap at 10
                elif field in ['entry_price']:
                    outcome_features.append(val / 10000.0)  # Normalize by typical ES price
                elif field in ['daily_loss_limit']:
                    outcome_features.append(val / 2000.0)  # Normalize by $2000
                elif field in ['wins_in_last_5_trades', 'losses_in_last_5_trades']:
                    outcome_features.append(val / 5.0)  # 0-1
                elif field in ['contracts']:
                    outcome_features.append(min(val / 5.0, 1.0))  # Cap at 5 contracts
                elif field in ['trade_number_in_session']:
                    outcome_features.append(min(val / 10.0, 1.0))  # Cap at 10 trades
                elif field in ['minutes_until_close']:
                    outcome_features.append(min(val / 480.0, 1.0))  # Cap at 8 hours
                elif field in ['commission_cost']:
                    outcome_features.append(min(val / 20.0, 1.0))  # Cap at $20
                elif field in ['slippage_ticks', 'volume_at_exit', 'bid_ask_spread_ticks']:
                    outcome_features.append(min(val / 10.0, 1.0))  # Generic normalization
                elif field in ['session', 'day_of_week']:
                    outcome_features.append(val / 7.0)  # 0-1
                elif field in ['entry_confidence']:
                    outcome_features.append(np.clip(val, 0, 1))  # Already 0-1
                elif field in ['vix']:
                    outcome_features.append(min(val / 50.0, 1.0))  # VIX 0-50
                else:
                    # Default: try to clip to 0-1 range
                    try:
                        outcome_features.append(np.clip(float(val), 0, 1))
                    except:
                        outcome_features.append(0.0)
            
            # Add categorical and summary features
            outcome_features.append(exit_reason_encoded)  # Exit reason (encoded)
            outcome_features.append(side_encoded)  # Side (0=long, 1=short)
            outcome_features.append(min(decision_count / 100.0, 1.0))  # Decision history count
            outcome_features.append(min(update_count / 50.0, 1.0))  # Exit param update count
            outcome_features.append(min(adjustment_count / 50.0, 1.0))  # Stop adjustment count
            outcome_features.append(min(pnl_history_count / 100.0, 1.0))  # P&L history count
            
            # EXIT PARAMS (131 features from EXIT_PARAMS config, excluding current_atr)
            exit_param_features = []
            # Use exit_params_config order to ensure consistency
            from exit_params_config import EXIT_PARAMS
            for param_name in EXIT_PARAMS.keys():
                val = exit_params_current.get(param_name, 0.0)
                # Normalize based on typical ranges
                if 'min_r' in param_name or 'target' in param_name or '_r' in param_name:
                    exit_param_features.append(np.clip(val / 10.0, 0, 1))  # R-multiples 0-10
                elif 'ticks' in param_name or 'distance' in param_name:
                    exit_param_features.append(np.clip(val / 50.0, 0, 1))  # Ticks 0-50
                elif 'multiplier' in param_name or 'atr' in param_name:
                    exit_param_features.append(np.clip(val / 5.0, 0, 1))  # ATR multiples 0-5
                elif 'bars' in param_name or 'timeout' in param_name:
                    exit_param_features.append(min(val / 200.0, 1.0))  # Bars 0-200
                elif 'threshold' in param_name or 'pct' in param_name:
                    exit_param_features.append(np.clip(val / 100.0, 0, 1))  # Percentages 0-100
                else:
                    exit_param_features.append(np.clip(val, 0, 1))  # Default 0-1
            
            # Add current_atr as 132nd feature (market context at time of trade)
            current_atr_val = exit_params_current.get('current_atr', market_state.get('atr', 2.0))
            exit_param_features.append(min(current_atr_val / 10.0, 1.0))
            
            # Combine all features (10 market + 63 outcome + 132 exit_params = 205 total)
            feature_vec = market_features + outcome_features + exit_param_features
            
            # Validate feature count
            if len(feature_vec) != 205:
                print(f"Warning: Expected 205 features, got {len(feature_vec)} - skipping experience")
                continue
            
            # Extract exit params (labels) - ALL 131 PARAMETERS
            # Use exit_params_used field (complete 131-param dict)
            exit_params_used = exp.get('exit_params_used', {})
            
            # If old data without exit_params_used, skip or use fallback
            if not exit_params_used:
                # Try old format (only 6 params)
                exit_params_used = exp.get('exit_params', {})
                if not exit_params_used:
                    continue  # Skip if no exit params at all
            
            # Normalize all 131 parameters to 0-1 range
            label_vec = normalize_exit_params(exit_params_used)
            
            # Validate we have 131 labels (not 6 from old model)
            if len(label_vec) == 131:
                features.append(feature_vec)
                labels.append(label_vec)
            else:
                print(f"Warning: Expected 131 exit params, got {len(label_vec)} - skipping experience")
                
        except Exception as e:
            print(f"Warning: Skipping experience due to error: {e}")
            continue
    
    print(f"Valid training samples: {len(features):,}")
    print()
    
    return np.array(features), np.array(labels)


def train_exit_model():
    """Train the exit parameters neural network"""
    
    # Load data
    X, y = load_exit_experiences()
    
    if len(X) < 100:
        print("ERROR: Need at least 100 exit experiences to train")
        return
    
    # Split train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Validation set: {len(X_val):,} samples")
    print()
    
    # Create datasets
    train_dataset = ExitDataset(X_train, y_train)
    val_dataset = ExitDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    print()
    
    # Model architecture: 205 inputs → 256 hidden → 256 hidden → 131 outputs
    model = ExitParamsNet(input_size=205, hidden_size=256).to(device)
    
    # Use MSE loss (predicting continuous values)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training loop
    print("Starting training...")
    print()
    
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    num_epochs = 150
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'exit_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_size': 205,
                'hidden_size': 256,
                'val_loss': val_loss
            }, model_path)
        else:
            patience_counter += 1
            if patience >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                print()
                break
    
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: data/exit_model.pth")
    print()
    print("Next steps:")
    print("  1. Integrate into local_exit_manager.py")
    print("  2. Run full_backtest.py to test improved exits")
    print("  3. Compare R-multiple and profit vs pattern matching")


if __name__ == '__main__':
    train_exit_model()
