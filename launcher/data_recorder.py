"""
Market Data Recorder
====================
Records live market data from brokers for backtesting purposes.

Captures:
- Quotes (Bid/Ask prices and sizes)
- Trades (Price, Size, Time)
- Market Depth/DOM (Order book levels)
- Timestamps

Output:
- Separate CSV file per symbol
- Append mode for continuous recording
- Chronologically ordered
"""

import csv
import time
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import threading
import sys

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from broker_interface import TopStepBroker
    from broker_websocket import BrokerWebSocketStreamer
    BROKER_AVAILABLE = True
except ImportError as e:
    BROKER_AVAILABLE = False
    BROKER_IMPORT_ERROR = str(e)

logger = logging.getLogger(__name__)

# Configuration constants
CSV_FLUSH_FREQUENCY = 100  # Flush CSV file every N records
STATS_REPORT_INTERVAL_SECONDS = 10  # Report statistics every N seconds


class MarketDataRecorder:
    """Records live market data to CSV for backtesting."""
    
    def __init__(
        self,
        broker: str,
        username: str,
        api_token: str,
        symbols: List[str],
        output_dir: str,
        log_callback: Optional[Callable[[str], None]] = None
    ):
        """
        Initialize market data recorder.
        
        Args:
            broker: Broker name (e.g., "TopStep")
            username: Broker username
            api_token: Broker API token
            symbols: List of symbols to record
            output_dir: Output directory for CSV files (one per symbol)
            log_callback: Optional callback for logging messages to GUI
        """
        self.broker_name = broker
        self.username = username
        self.api_token = api_token
        self.symbols = symbols
        self.output_dir = Path(output_dir)
        self.log_callback = log_callback
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Broker connection
        self.broker = None
        self.websocket = None
        
        # Recording state
        self.is_recording = False
        
        # Per-symbol CSV files and writers
        self.csv_files = {}  # symbol -> file handle
        self.csv_writers = {}  # symbol -> csv.writer
        self.csv_locks = {symbol: threading.Lock() for symbol in symbols}
        
        # Statistics
        self.stats = {symbol: {
            'quotes': 0,
            'trades': 0,
            'depth_updates': 0
        } for symbol in symbols}
        
        # Contract ID mapping (symbol -> contract_id)
        self.contract_ids = {}
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        return datetime.now().isoformat()
    
    def log(self, message: str):
        """Log message to callback and logger."""
        logger.info(message)
        if self.log_callback:
            self.log_callback(message)
    
    def start(self):
        """Start recording market data."""
        # Check if broker modules are available
        if not BROKER_AVAILABLE:
            raise Exception(
                f"Broker SDK not available: {BROKER_IMPORT_ERROR}\n\n"
                "Please install broker dependencies by uncommenting the broker SDK section "
                "in requirements.txt and running:\n"
                "pip install -r requirements.txt\n\n"
                "Required packages:\n"
                "- project-x-py>=3.5.9\n"
                "- signalrcore>=0.9.5\n"
                "- And other broker dependencies listed in requirements.txt"
            )
        
        try:
            self.log("Connecting to broker...")
            
            # Connect to broker
            if self.broker_name == "TopStep":
                self.broker = TopStepBroker(
                    api_token=self.api_token,
                    username=self.username
                )
                
                if not self.broker.connect():
                    raise Exception("Failed to connect to broker")
                
                self.log("✓ Connected to broker")
                
                # Get contract IDs for symbols
                self.log("Looking up contract IDs...")
                for symbol in self.symbols:
                    try:
                        contract_id = self.broker.get_contract_id(symbol)
                        if contract_id:
                            self.contract_ids[symbol] = contract_id
                            self.log(f"✓ {symbol} -> Contract ID: {contract_id}")
                        else:
                            self.log(f"⚠ Warning: Could not find contract ID for {symbol}")
                    except Exception as e:
                        self.log(f"⚠ Error getting contract ID for {symbol}: {e}")
                
                if not self.contract_ids:
                    raise Exception("No valid contract IDs found for selected symbols")
                
                # Connect to WebSocket
                self.log("Connecting to market data stream...")
                session_token = self.broker.session_token
                self.websocket = BrokerWebSocketStreamer(
                    session_token=session_token,
                    hub_url="wss://rtc.topstepx.com/hubs/market"
                )
                
                if not self.websocket.connect():
                    raise Exception("Failed to connect to WebSocket")
                
                self.log("✓ Connected to market data stream")
                
                # Initialize CSV files
                self.log(f"Initializing CSV files in: {self.output_dir}")
                self._initialize_csv()
                
                # Subscribe to market data for each symbol
                self.is_recording = True
                for symbol, contract_id in self.contract_ids.items():
                    self.log(f"Subscribing to {symbol} market data...")
                    
                    # Subscribe to quotes
                    self.websocket.subscribe_quotes(
                        contract_id,
                        lambda data, sym=symbol: self._on_quote(sym, data)
                    )
                    
                    # Subscribe to trades
                    self.websocket.subscribe_trades(
                        contract_id,
                        lambda data, sym=symbol: self._on_trade(sym, data)
                    )
                    
                    # Subscribe to depth/DOM
                    try:
                        self.websocket.subscribe_depth(
                            contract_id,
                            lambda data, sym=symbol: self._on_depth(sym, data)
                        )
                    except Exception as e:
                        self.log(f"⚠ Could not subscribe to depth for {symbol}: {e}")
                    
                    self.log(f"✓ Subscribed to {symbol}")
                
                self.log("=" * 50)
                self.log("RECORDING STARTED")
                self.log(f"Recording {len(self.contract_ids)} symbols: {', '.join(self.contract_ids.keys())}")
                self.log(f"Output directory: {self.output_dir}")
                self.log("=" * 50)
                
                # Start statistics reporter
                self._start_stats_reporter()
                
            else:
                raise Exception(f"Unsupported broker: {self.broker_name}")
                
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.stop()
            raise
    
    def _initialize_csv(self):
        """Initialize CSV files for each symbol with headers (append mode)."""
        headers = [
            'timestamp',
            'data_type',  # quote, trade, or depth
            'bid_price',
            'bid_size',
            'ask_price',
            'ask_size',
            'trade_price',
            'trade_size',
            'trade_side',  # buy or sell
            'depth_level',
            'depth_side',  # bid or ask
            'depth_price',
            'depth_size'
        ]
        
        for symbol in self.symbols:
            csv_path = self.output_dir / f"{symbol}.csv"
            
            # Check if file exists to determine if we need to write headers
            file_exists = csv_path.exists()
            
            # Open in append mode to continue from where we left off
            self.csv_files[symbol] = open(csv_path, 'a', newline='')
            self.csv_writers[symbol] = csv.writer(self.csv_files[symbol])
            
            # Write header only if file is new
            if not file_exists:
                self.csv_writers[symbol].writerow(headers)
                self.csv_files[symbol].flush()
                self.log(f"✓ Created new CSV file: {csv_path}")
            else:
                self.log(f"✓ Appending to existing CSV file: {csv_path}")
    
    def _write_csv_row(self, symbol: str, data: Dict[str, Any]):
        """Write a row to the symbol's CSV file (thread-safe)."""
        with self.csv_locks[symbol]:
            if symbol in self.csv_writers and self.is_recording:
                row = [
                    data.get('timestamp', ''),
                    data.get('data_type', ''),
                    data.get('bid_price', ''),
                    data.get('bid_size', ''),
                    data.get('ask_price', ''),
                    data.get('ask_size', ''),
                    data.get('trade_price', ''),
                    data.get('trade_size', ''),
                    data.get('trade_side', ''),
                    data.get('depth_level', ''),
                    data.get('depth_side', ''),
                    data.get('depth_price', ''),
                    data.get('depth_size', '')
                ]
                self.csv_writers[symbol].writerow(row)
                
                # Flush periodically to ensure data is written
                if sum(self.stats[symbol].values()) % CSV_FLUSH_FREQUENCY == 0:
                    self.csv_files[symbol].flush()
    
    def _on_quote(self, symbol: str, data: Any):
        """Handle quote data."""
        if not self.is_recording:
            return
        
        try:
            # Extract quote data
            # Data structure may vary by broker, adjust as needed
            timestamp = self._get_current_timestamp()
            
            # Try to extract bid/ask from data object
            bid_price = getattr(data, 'bid_price', None) or getattr(data, 'Bid', None)
            bid_size = getattr(data, 'bid_size', None) or getattr(data, 'BidSize', None)
            ask_price = getattr(data, 'ask_price', None) or getattr(data, 'Ask', None)
            ask_size = getattr(data, 'ask_size', None) or getattr(data, 'AskSize', None)
            
            # If data is a list/dict, try to access it that way
            if isinstance(data, (list, tuple)) and len(data) >= 1:
                quote_obj = data[0]
                bid_price = getattr(quote_obj, 'bid_price', None) or getattr(quote_obj, 'Bid', None)
                bid_size = getattr(quote_obj, 'bid_size', None) or getattr(quote_obj, 'BidSize', None)
                ask_price = getattr(quote_obj, 'ask_price', None) or getattr(quote_obj, 'Ask', None)
                ask_size = getattr(quote_obj, 'ask_size', None) or getattr(quote_obj, 'AskSize', None)
            
            row_data = {
                'timestamp': timestamp,
                'data_type': 'quote',
                'bid_price': bid_price or '',
                'bid_size': bid_size or '',
                'ask_price': ask_price or '',
                'ask_size': ask_size or ''
            }
            
            self._write_csv_row(symbol, row_data)
            self.stats[symbol]['quotes'] += 1
            
        except Exception as e:
            logger.error(f"Error processing quote for {symbol}: {e}")
    
    def _on_trade(self, symbol: str, data: Any):
        """Handle trade data."""
        if not self.is_recording:
            return
        
        try:
            timestamp = self._get_current_timestamp()
            
            # Try to extract trade data
            trade_price = getattr(data, 'price', None) or getattr(data, 'Price', None)
            trade_size = getattr(data, 'size', None) or getattr(data, 'Size', None)
            trade_side = getattr(data, 'side', None) or getattr(data, 'Side', None)
            
            # If data is a list/dict, try to access it that way
            if isinstance(data, (list, tuple)) and len(data) >= 1:
                trade_obj = data[0]
                trade_price = getattr(trade_obj, 'price', None) or getattr(trade_obj, 'Price', None)
                trade_size = getattr(trade_obj, 'size', None) or getattr(trade_obj, 'Size', None)
                trade_side = getattr(trade_obj, 'side', None) or getattr(trade_obj, 'Side', None)
            
            row_data = {
                'timestamp': timestamp,
                'data_type': 'trade',
                'trade_price': trade_price or '',
                'trade_size': trade_size or '',
                'trade_side': trade_side or ''
            }
            
            self._write_csv_row(symbol, row_data)
            self.stats[symbol]['trades'] += 1
            
        except Exception as e:
            logger.error(f"Error processing trade for {symbol}: {e}")
    
    def _on_depth(self, symbol: str, data: Any):
        """Handle market depth/DOM data."""
        if not self.is_recording:
            return
        
        try:
            timestamp = self._get_current_timestamp()
            
            # Market depth is typically an array of price levels
            # Data structure may vary by broker
            # Try to extract bid and ask levels
            
            # Process as list of levels
            if isinstance(data, (list, tuple)):
                for i, level in enumerate(data):
                    # Try to extract level data
                    side = getattr(level, 'side', None) or getattr(level, 'Side', None)
                    price = getattr(level, 'price', None) or getattr(level, 'Price', None)
                    size = getattr(level, 'size', None) or getattr(level, 'Size', None)
                    
                    if price is not None:
                        row_data = {
                            'timestamp': timestamp,
                            'data_type': 'depth',
                            'depth_level': i,
                            'depth_side': side or '',
                            'depth_price': price or '',
                            'depth_size': size or ''
                        }
                        self._write_csv_row(symbol, row_data)
            
            self.stats[symbol]['depth_updates'] += 1
            
        except Exception as e:
            logger.error(f"Error processing depth for {symbol}: {e}")
    
    def _start_stats_reporter(self):
        """Start background thread to report statistics."""
        def report_stats():
            while self.is_recording:
                time.sleep(STATS_REPORT_INTERVAL_SECONDS)
                if self.is_recording:
                    total_quotes = sum(s['quotes'] for s in self.stats.values())
                    total_trades = sum(s['trades'] for s in self.stats.values())
                    total_depth = sum(s['depth_updates'] for s in self.stats.values())
                    
                    self.log(
                        f"Stats: Quotes={total_quotes}, Trades={total_trades}, "
                        f"Depth Updates={total_depth}"
                    )
        
        stats_thread = threading.Thread(target=report_stats, daemon=True)
        stats_thread.start()
    
    def stop(self):
        """Stop recording and cleanup."""
        self.log("Stopping recorder...")
        self.is_recording = False
        
        # Close all CSV files
        for symbol, csv_file in self.csv_files.items():
            try:
                with self.csv_locks[symbol]:
                    csv_file.flush()
                    csv_file.close()
                csv_path = self.output_dir / f"{symbol}.csv"
                self.log(f"✓ CSV file saved: {csv_path}")
            except Exception as e:
                self.log(f"⚠ Error closing file for {symbol}: {e}")
        
        # Disconnect WebSocket
        if self.websocket:
            try:
                self.websocket.disconnect()
                self.log("✓ Disconnected from market data stream")
            except:
                pass
        
        # Disconnect broker
        if self.broker:
            try:
                self.broker.disconnect()
                self.log("✓ Disconnected from broker")
            except:
                pass
        
        # Print final statistics
        self.log("=" * 50)
        self.log("RECORDING STOPPED")
        self.log("Final Statistics:")
        for symbol, stats in self.stats.items():
            self.log(
                f"  {symbol}: Quotes={stats['quotes']}, "
                f"Trades={stats['trades']}, Depth={stats['depth_updates']}"
            )
        self.log("=" * 50)
