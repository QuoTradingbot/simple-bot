"""
Hybrid Model Comparison Script
===============================
Runs 10-day backtests comparing:
1. Original Neural Network Model
2. Hybrid V1: Pattern-Matching Confidence Booster
3. Hybrid V2: Adaptive Threshold Model

Generates comprehensive comparison report.
"""

import sys
import os
import json
from datetime import datetime

# Add paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, '..', 'src'))

def run_backtest_with_model(model_name, model_instance, days=10):
    """
    Run backtest with a specific model.
    
    Args:
        model_name: Name of the model
        model_instance: Instance of the confidence model
        days: Number of days to backtest
    
    Returns:
        Dict of results
    """
    print(f"\n{'='*80}")
    print(f"RUNNING BACKTEST: {model_name}")
    print(f"{'='*80}\n")
    
    # Import backtest function
    # Note: We'll need to modify full_backtest.py to accept a model instance
    # For now, we'll create a simplified version
    
    from full_backtest import run_full_backtest
    
    # Run backtest (this will use the global config)
    # We'll need to inject our model somehow
    # For simplicity, let's just import and modify the config
    
    import full_backtest as bt
    
    # Save original local_manager
    original_manager = bt.local_manager
    
    try:
        # Replace with our model
        bt.local_manager = model_instance
        
        # Run backtest
        csv_file = os.path.join(script_dir, "..", "data", "historical_data", "ES_1min_cleaned.csv")
        df_trades = run_full_backtest(csv_file, days=days)
        
        # Extract results
        if df_trades is not None and len(df_trades) > 0:
            total_trades = len(df_trades)
            winning_trades = len(df_trades[df_trades['pnl'] > 0])
            losing_trades = len(df_trades[df_trades['pnl'] <= 0])
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = df_trades['pnl'].sum()
            
            results = {
                'model_name': model_name,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': df_trades['pnl'].mean(),
                'max_win': df_trades['pnl'].max(),
                'max_loss': df_trades['pnl'].min(),
                'trades_df': df_trades
            }
        else:
            results = {
                'model_name': model_name,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'avg_pnl': 0,
                'max_win': 0,
                'max_loss': 0,
                'trades_df': None
            }
        
        # Add model-specific stats
        if hasattr(model_instance, 'get_stats'):
            results['model_stats'] = model_instance.get_stats()
        
        return results
        
    finally:
        # Restore original manager
        bt.local_manager = original_manager


def print_comparison_report(results_list):
    """
    Print comprehensive comparison report.
    
    Args:
        results_list: List of result dictionaries
    """
    print(f"\n{'='*80}")
    print(f"HYBRID MODEL COMPARISON REPORT")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<30} | {'Trades':>8} | {'Win%':>6} | {'Total P&L':>12} | {'Avg P&L':>10}")
    print(f"{'-'*80}")
    
    for results in results_list:
        print(f"{results['model_name']:<30} | "
              f"{results['total_trades']:>8} | "
              f"{results['win_rate']:>6.1f}% | "
              f"${results['total_pnl']:>11.2f} | "
              f"${results['avg_pnl']:>9.2f}")
    
    print(f"\n{'='*80}")
    print(f"DETAILED ANALYSIS")
    print(f"{'='*80}\n")
    
    for results in results_list:
        print(f"\n{results['model_name']}:")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Wins: {results['winning_trades']} | Losses: {results['losing_trades']}")
        print(f"  Win Rate: {results['win_rate']:.1f}%")
        print(f"  Total P&L: ${results['total_pnl']:.2f}")
        print(f"  Average P&L: ${results['avg_pnl']:.2f}")
        print(f"  Max Win: ${results['max_win']:.2f}")
        print(f"  Max Loss: ${results['max_loss']:.2f}")
        
        if 'model_stats' in results:
            print(f"\n  Model-Specific Stats:")
            stats = results['model_stats']
            if 'hybrid_v1' in stats:
                print(f"    Hybrid V1 Boosts: {stats['hybrid_v1']['boosts_applied']} "
                      f"({stats['hybrid_v1']['boost_rate']})")
            if 'hybrid_v2' in stats:
                print(f"    Hybrid V2 Threshold: {stats['hybrid_v2']['current_threshold']}")
                print(f"    Adjustments: {stats['hybrid_v2']['threshold_adjustments']}")
    
    # Determine best model
    print(f"\n{'='*80}")
    print(f"RECOMMENDATION")
    print(f"{'='*80}\n")
    
    best_pnl = max(results_list, key=lambda x: x['total_pnl'])
    best_winrate = max(results_list, key=lambda x: x['win_rate'])
    most_trades = max(results_list, key=lambda x: x['total_trades'])
    
    print(f"🏆 Best P&L: {best_pnl['model_name']} (${best_pnl['total_pnl']:.2f})")
    print(f"🎯 Best Win Rate: {best_winrate['model_name']} ({best_winrate['win_rate']:.1f}%)")
    print(f"📊 Most Active: {most_trades['model_name']} ({most_trades['total_trades']} trades)")
    
    # Overall recommendation
    if best_pnl['total_trades'] >= 5:  # Need minimum trades
        print(f"\n✅ OVERALL RECOMMENDATION: {best_pnl['model_name']}")
        print(f"   This model achieved the best P&L with sufficient trade sample size.")
    else:
        print(f"\n⚠️  WARNING: All models have low trade counts. Consider:")
        print(f"   - Adjusting confidence thresholds")
        print(f"   - Retraining neural network")
        print(f"   - Using longer backtest period")


def main():
    """Main execution."""
    print(f"🔬 HYBRID MODEL COMPARISON TEST")
    print(f"Testing 3 models on 10-day backtest")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # We can't easily modify full_backtest.py without changing the core logic
    # So let's create a simpler comparison by running the script 3 times with different configs
    
    print("⚠️  NOTE: This script requires modification of full_backtest.py to inject models.")
    print("For now, please run the backtest manually with each model:")
    print()
    print("1. Original Model (current system):")
    print("   cd dev-tools && python full_backtest.py --days 10 --no-save")
    print()
    print("2. Hybrid V1 (Pattern-Matching Booster):")
    print("   Modify full_backtest.py line 94-106 to use hybrid_model_v1.HybridConfidenceV1")
    print()
    print("3. Hybrid V2 (Adaptive Threshold):")
    print("   Modify full_backtest.py line 94-106 to use hybrid_model_v2.HybridConfidenceV2")
    print()
    print("📝 Results will be saved to /tmp/hybrid_comparison_results.json")


if __name__ == "__main__":
    main()
