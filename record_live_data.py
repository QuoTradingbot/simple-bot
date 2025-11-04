"""
Example: Record Live ES Data with Bid/Ask and Order Execution
Run this alongside your bot to capture all market data and order placements
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from live_data_recorder import LiveDataRecorder
from broker_interface import TopStepBroker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def record_live_data(
    symbol: str = "ES",
    duration_minutes: int = 60,
    record_interval_seconds: float = 0.1  # 10 ticks per second
):
    """
    Record live market data from TopStep
    
    Args:
        symbol: Contract symbol (ES, NQ, etc.)
        duration_minutes: How long to record (0 = infinite)
        record_interval_seconds: Time between tick captures
    """
    logger.info("=" * 60)
    logger.info("LIVE DATA RECORDER")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Duration: {duration_minutes} minutes" if duration_minutes > 0 else "Duration: Infinite (Ctrl+C to stop)")
    logger.info(f"Sampling rate: {1/record_interval_seconds:.1f} ticks/second")
    logger.info("=" * 60)
    
    # Initialize recorder
    recorder = LiveDataRecorder(
        output_dir="live_data_recordings",
        symbol=symbol,
        compress=True,  # Save space with gzip
        max_file_size_mb=100,  # Rotate at 100MB
        rotation_interval_minutes=60  # Rotate every hour
    )
    
    # Connect to TopStep
    logger.info("Connecting to TopStep...")
    broker = TopStepBroker()
    
    if not broker.connect():
        logger.error("Failed to connect to TopStep - exiting")
        return
    
    logger.info("Connected! Starting data recording...")
    
    # Get contract ID for symbol
    contract_id = broker.get_contract_id(symbol)
    if not contract_id:
        logger.error(f"Could not find contract ID for {symbol}")
        broker.disconnect()
        return
    
    logger.info(f"Contract ID: {contract_id}")
    
    # Recording loop
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60) if duration_minutes > 0 else float('inf')
    
    last_stats_time = start_time
    stats_interval = 60  # Print stats every 60 seconds
    
    try:
        while time.time() < end_time:
            # Get current quote
            quote = broker.get_quote(symbol)
            
            if quote:
                # Extract bid/ask/last
                bid = getattr(quote, 'bid', None) or getattr(quote, 'bid_price', 0)
                ask = getattr(quote, 'ask', None) or getattr(quote, 'ask_price', 0)
                last = getattr(quote, 'last', None) or getattr(quote, 'last_price', 0)
                bid_size = getattr(quote, 'bid_size', 0)
                ask_size = getattr(quote, 'ask_size', 0)
                volume = getattr(quote, 'volume', 0)
                
                if bid > 0 and ask > 0:
                    # Record the tick
                    recorder.record_tick(
                        bid=bid,
                        ask=ask,
                        last=last,
                        bid_size=bid_size,
                        ask_size=ask_size,
                        volume=volume
                    )
            
            # Print stats periodically
            if time.time() - last_stats_time > stats_interval:
                stats = recorder.get_stats()
                logger.info(f"Stats: {stats['ticks_recorded']} ticks recorded "
                           f"({stats['ticks_per_second']} ticks/sec) - "
                           f"File: {stats['current_tick_file']} ({stats['tick_file_size_mb']} MB)")
                last_stats_time = time.time()
            
            # Sleep before next sample
            time.sleep(record_interval_seconds)
    
    except KeyboardInterrupt:
        logger.info("\n[STOPPED] Recording stopped by user")
    
    except Exception as e:
        logger.error(f"Error during recording: {e}", exc_info=True)
    
    finally:
        # Cleanup
        logger.info("Shutting down...")
        recorder.close()
        broker.disconnect()
        logger.info("Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Record live ES market data")
    parser.add_argument("--symbol", default="ES", help="Trading symbol (default: ES)")
    parser.add_argument("--duration", type=int, default=0, help="Recording duration in minutes (0 = infinite)")
    parser.add_argument("--rate", type=float, default=0.1, help="Seconds between samples (default: 0.1 = 10/sec)")
    
    args = parser.parse_args()
    
    record_live_data(
        symbol=args.symbol,
        duration_minutes=args.duration,
        record_interval_seconds=args.rate
    )
