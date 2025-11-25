"""
Backtest Reporter - Live progress and results display
"""
import time
from datetime import datetime
from typing import Optional, Dict, Any


class BacktestReporter:
    """Reports backtest progress and results in real-time"""
    
    def __init__(self, starting_balance: float = 50000.0):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.trades = []
        self.signals_approved = 0
        self.signals_rejected = 0
        self.total_bars = 0
        self.start_time = time.time()
        
    def print_header(self, start_date=None, end_date=None, config=None, symbol="ES"):
        """Print backtest header"""
        print("\n" + "="*80)
        print("BACKTEST STARTING")
        print("="*80)
        print(f"Symbol: {symbol}")
        if start_date and end_date:
            print(f"Period: {start_date} to {end_date}")
        print(f"Starting Balance: ${self.starting_balance:,.2f}")
        if config:
            print(f"Max Contracts: {config.get('max_contracts', 1)}")
            print(f"RL Exploration: {config.get('rl_exploration_rate', 0.0)}")
        print("="*80 + "\n")
        
    def record_signal(self, approved: bool):
        """Record signal decision"""
        if approved:
            self.signals_approved += 1
        else:
            self.signals_rejected += 1
            
    def record_trade(self, trade: Dict[str, Any]):
        """Record completed trade"""
        self.trades.append(trade)
        self.current_balance += trade.get('pnl', 0)
        
        # Print trade result
        pnl = trade.get('pnl', 0)
        symbol = "Γ£ô" if pnl > 0 else "Γ£ù"
        print(f"{symbol} Trade #{len(self.trades)}: {trade.get('side', 'N/A').upper()} | "
              f"P&L: ${pnl:+,.2f} | Balance: ${self.current_balance:,.2f}")
        
    def update_progress(self, bars_processed: int, total_bars: int):
        """Update progress bar"""
        if total_bars > 0 and bars_processed % 1000 == 0:
            pct = (bars_processed / total_bars) * 100
            print(f"Progress: {bars_processed:,}/{total_bars:,} bars ({pct:.1f}%)")
            
    def print_summary(self):
        """Print final backtest summary"""
        elapsed = time.time() - self.start_time
        
        wins = [t for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t for t in self.trades if t.get('pnl', 0) < 0]
        
        total_pnl = self.current_balance - self.starting_balance
        win_rate = (len(wins) / len(self.trades) * 100) if self.trades else 0
        
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        
        print("\n" + "="*80)
        print("BACKTEST COMPLETE")
        print("="*80)
        print(f"Trades: {len(self.trades)} (Wins: {len(wins)}, Losses: {len(losses)})")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Starting Balance: ${self.starting_balance:,.2f}")
        print(f"Ending Balance: ${self.current_balance:,.2f}")
        print(f"Total P&L: ${total_pnl:+,.2f} ({total_pnl/self.starting_balance*100:+.2f}%)")
        print(f"Avg Win: ${avg_win:,.2f}")
        print(f"Avg Loss: ${avg_loss:,.2f}")
        print(f"Signals: {self.signals_approved} approved, {self.signals_rejected} rejected")
        print(f"Execution Time: {elapsed:.1f}s")
        print("="*80 + "\n")


# Global reporter instance
_reporter: Optional[BacktestReporter] = None


def get_reporter() -> BacktestReporter:
    """Get or create reporter instance"""
    global _reporter
    if _reporter is None:
        _reporter = BacktestReporter()
    return _reporter


def reset_reporter(starting_balance: float = 50000.0) -> BacktestReporter:
    """Reset reporter for new backtest"""
    global _reporter
    _reporter = BacktestReporter(starting_balance)
    return _reporter
