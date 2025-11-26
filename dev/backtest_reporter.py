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
        """Print backtest header with comprehensive configuration"""
        print("\n" + "="*80)
        print("                        BACKTEST CONFIGURATION")
        print("="*80)
        print(f"Symbol:           {symbol}")
        if start_date and end_date:
            print(f"Period:           {start_date} to {end_date}")
        print(f"Starting Balance: ${self.starting_balance:,.2f}")
        print(f"Max Contracts:    {self.max_contracts}")
        if config:
            # Show key trading parameters from config.json
            print(f"\nTrading Parameters (from config.json):")
            print(f"  Risk Per Trade:         {config.get('risk_per_trade', 0.01)*100:.1f}%")
            print(f"  Daily Loss Limit:       ${config.get('daily_loss_limit', 0):,.2f}")
            print(f"  RL Exploration Rate:    {config.get('rl_exploration_rate', 0.0)*100:.1f}%")
            print(f"  RL Confidence Threshold: {config.get('rl_confidence_threshold', 0.0)*100:.1f}%")
            
            # Show filters if enabled
            filters = []
            if config.get('use_rsi_filter', False):
                filters.append(f"RSI({config.get('rsi_period', 14)})")
            if config.get('use_vwap_direction_filter', False):
                filters.append("VWAP Direction")
            if config.get('use_trend_filter', False):
                filters.append("Trend")
            if config.get('use_volume_filter', False):
                filters.append("Volume")
            
            if filters:
                print(f"  Active Filters:         {', '.join(filters)}")
        
        print("="*80)
        print("                           TRADE LOG")
        print("="*80 + "\n")
        
    def record_signal(self, approved: bool):
        """Record signal decision"""
        if approved:
            self.signals_approved += 1
        else:
            self.signals_rejected += 1
            
    def record_trade(self, trade: Dict[str, Any]):
        """Record completed trade with clean, informative output"""
        self.trades.append(trade)
        self.current_balance += trade.get('pnl', 0)
        
        # Get trade details
        pnl = trade.get('pnl', 0)
        side = trade.get('side', 'N/A').upper()
        
        # Determine win/loss status based on actual P&L
        if pnl > 0:
            status = "[WIN] "
        elif pnl < 0:
            status = "[LOSS]"
        else:
            status = "[B/E] "  # Break even
        
        # Format entry/exit times
        entry_time = trade.get('entry_time', '')
        exit_time = trade.get('exit_time', '')
        
        # Handle different time formats (datetime objects or strings)
        if hasattr(entry_time, 'strftime'):
            entry_str = entry_time.strftime('%a %m/%d %H:%M')
            exit_str = exit_time.strftime('%H:%M') if hasattr(exit_time, 'strftime') else 'N/A'
        elif isinstance(entry_time, str) and len(entry_time) >= 10:
            try:
                dt = datetime.fromisoformat(entry_time.replace('Z', '+00:00') if 'Z' in entry_time else entry_time)
                entry_str = dt.strftime('%a %m/%d %H:%M')
                if isinstance(exit_time, str):
                    dt_exit = datetime.fromisoformat(exit_time.replace('Z', '+00:00') if 'Z' in exit_time else exit_time)
                    exit_str = dt_exit.strftime('%H:%M')
                else:
                    exit_str = 'N/A'
            except:
                entry_str = entry_time[:16] if entry_time else 'N/A'
                exit_str = 'N/A'
        else:
            entry_str = 'N/A'
            exit_str = 'N/A'
        
        # Get trade details
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        
        # Calculate duration from entry/exit times if duration_minutes is wrong
        duration = trade.get('duration_minutes', 0)
        if duration > 10000 or duration < 0:  # Unrealistic duration
            if hasattr(entry_time, 'timestamp') and hasattr(exit_time, 'timestamp'):
                duration = (exit_time - entry_time).total_seconds() / 60.0
        duration = abs(duration)
        
        confidence = trade.get('confidence', 0)
        exit_reason = trade.get('exit_reason', 'unknown')
        qty = self.max_contracts
        
        # Get regime info if available
        regime = trade.get('regime', '')
        regime_str = f" | {regime}" if regime else ""
        
        # Clean output format:
        # ✓ WIN : LONG 1x | Entry: Mon 11/25 14:30 @ $5250.00 -> Exit: 14:45 @ $5255.00 | P&L: $+25.00 | target | 15min | Conf: 85%
        print(f"{status}: {side:5} {qty}x | Entry: {entry_str} @ ${entry_price:.2f} -> Exit: {exit_str} @ ${exit_price:.2f} | "
              f"P&L: ${pnl:+8.2f} | {exit_reason:12} | {duration:3.0f}min | Conf: {confidence:3.0f}%{regime_str}")
        
    def update_progress(self, bars_processed: int, total_bars: int):
        """Update progress - only show every 10% to reduce spam"""
        if total_bars > 0:
            pct = (bars_processed / total_bars) * 100
            # Only show progress every 10% or at completion
            if bars_processed == total_bars or bars_processed % max(1, total_bars // 10) == 0:
                # Clean progress format
                print(f"Progress: [{pct:5.1f}%] {bars_processed:,}/{total_bars:,} bars processed", end='\r')
                if bars_processed == total_bars:
                    print()  # New line when complete
            
    def print_summary(self):
        """Print comprehensive backtest summary"""
        elapsed = time.time() - self.start_time
        
        wins = [t for t in self.trades if t.get('pnl', 0) > 0]
        losses = [t for t in self.trades if t.get('pnl', 0) < 0]
        breakevens = [t for t in self.trades if t.get('pnl', 0) == 0]
        
        total_pnl = self.current_balance - self.starting_balance
        win_rate = (len(wins) / len(self.trades) * 100) if self.trades else 0
        
        avg_win = sum(t['pnl'] for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t['pnl'] for t in losses) / len(losses) if losses else 0
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in wins)
        gross_loss = abs(sum(t['pnl'] for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate max drawdown
        max_dd = 0
        peak = self.starting_balance
        for t in self.trades:
            balance = self.starting_balance + sum(tr['pnl'] for tr in self.trades[:self.trades.index(t)+1])
            if balance > peak:
                peak = balance
            drawdown = peak - balance
            if drawdown > max_dd:
                max_dd = drawdown
        
        max_dd_pct = (max_dd / peak * 100) if peak > 0 else 0
        
        # Calculate average trade duration
        avg_duration = sum(t.get('duration_minutes', 0) for t in self.trades) / len(self.trades) if self.trades else 0
        
        print("\n" + "="*80)
        print("                          BACKTEST SUMMARY")
        print("="*80)
        print(f"\nPerformance:")
        print(f"  Total Trades:      {len(self.trades)} (Wins: {len(wins)}, Losses: {len(losses)}, B/E: {len(breakevens)})")
        print(f"  Win Rate:          {win_rate:.1f}%")
        print(f"  Profit Factor:     {profit_factor:.2f}")
        print(f"  Avg Trade Duration: {avg_duration:.1f} minutes")
        
        print(f"\nP&L Analysis:")
        print(f"  Starting Balance:  ${self.starting_balance:,.2f}")
        print(f"  Ending Balance:    ${self.current_balance:,.2f}")
        print(f"  Net P&L:           ${total_pnl:+,.2f} ({total_pnl/self.starting_balance*100:+.2f}%)")
        print(f"  Avg Win:           ${avg_win:,.2f}")
        print(f"  Avg Loss:          ${avg_loss:,.2f}")
        print(f"  Largest Win:       ${max([t['pnl'] for t in wins], default=0):,.2f}")
        print(f"  Largest Loss:      ${min([t['pnl'] for t in losses], default=0):,.2f}")
        
        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:      ${max_dd:,.2f} ({max_dd_pct:.2f}%)")
        
        print(f"\nSignal Performance:")
        total_signals = self.signals_approved + self.signals_rejected
        print(f"  Total Signals:     {total_signals}")
        print(f"  Trades Taken:      {len(self.trades)}")
        
        print(f"\nExecution:")
        print(f"  Total Bars:        {self.total_bars:,}")
        print(f"  Execution Time:    {elapsed:.1f}s")
        
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
