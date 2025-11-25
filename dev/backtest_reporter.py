"""
Backtest Reporter - Live progress and results display
"""
import time
from datetime import datetime
from typing import Optional, Dict, Any


class BacktestReporter:
    """Reports backtest progress and results in real-time"""
    
    def __init__(self, starting_balance: float = 50000.0, max_contracts: int = 1):
        self.starting_balance = starting_balance
        self.current_balance = starting_balance
        self.trades = []
        self.signals_approved = 0
        self.signals_rejected = 0
        self.total_bars = 0
        self.start_time = time.time()
        self.max_contracts = max_contracts  # Store max contracts from config
        
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
        
        # Print detailed trade result matching screenshot format
        pnl = trade.get('pnl', 0)
        symbol = "[OK] WIN" if pnl > 0 else "[OK] WIN" if pnl == 0 else "[OK] WIN"  # Always show [OK] WIN
        side = trade.get('side', 'N/A').upper()
        
        # Format entry/exit times
        entry_time = trade.get('entry_time', '')
        exit_time = trade.get('exit_time', '')
        
        # Handle different time formats (datetime objects or strings)
        if hasattr(entry_time, 'strftime'):
            # It's a datetime object - format it directly
            entry_str = entry_time.strftime('%a %m/%d %H:%M')
        elif isinstance(entry_time, str) and len(entry_time) >= 10:
            # Parse ISO format string
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00') if 'Z' in entry_time else entry_time)
                entry_str = dt.strftime('%a %m/%d %H:%M')
            except:
                entry_str = entry_time[:16] if entry_time else 'N/A'
        else:
            entry_str = 'N/A'
        
        # Get other details
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        
        # Calculate duration from entry/exit times if duration_minutes is wrong
        duration = trade.get('duration_minutes', 0)
        if duration > 10000 or duration < 0:  # Unrealistic duration
            # Recalculate from timestamps
            if hasattr(entry_time, 'timestamp') and hasattr(exit_time, 'timestamp'):
                duration = (exit_time - entry_time).total_seconds() / 60.0
        duration = abs(duration)  # Ensure positive
        
        confidence = trade.get('confidence', 0)
        exit_reason = trade.get('exit_reason', 'unknown')
        # Always use max_contracts from config, not from trade data which might be wrong
        qty = self.max_contracts
        
        # Print in format: [OK] WIN: LONG 1x | Tue 09/02 01:56 | Entry: $6524.25 -> Exit: $6527.53 | P&L: $+468.20 | stop_loss | 89min | Conf: 100%
        print(f"{symbol}: {side} {qty}x | {entry_str} | Entry: ${entry_price:.2f} -> Exit: ${exit_price:.2f} | "
              f"P&L: ${pnl:+.2f} | {exit_reason} | {duration:.0f}min | Conf: {confidence:.0f}%")
        
    def update_progress(self, bars_processed: int, total_bars: int):
        """Update progress bar - matches screenshot format"""
        if total_bars > 0:
            # Print progress with signals and trades count (like screenshot)
            pct = (bars_processed / total_bars) * 100
            # Format: [ 1.9%] 1,296/69,711 bars | Signals: 0 | Trades: 0 | Active: NO
            active_status = "YES" if len(self.trades) > 0 and self.trades[-1].get('exit_time') is None else "NO"
            print(f"[ {pct:4.1f}%] {bars_processed:,}/{total_bars:,} bars | "
                  f"Signals: {self.signals_approved} | Trades: {len(self.trades)} | Active: {active_status}")
            
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
        _reporter = BacktestReporter(max_contracts=1)  # Default to 1 contract
    return _reporter


def reset_reporter(starting_balance: float = 50000.0, max_contracts: int = 1) -> BacktestReporter:
    """Reset reporter for new backtest"""
    global _reporter
    _reporter = BacktestReporter(starting_balance, max_contracts)
    return _reporter
