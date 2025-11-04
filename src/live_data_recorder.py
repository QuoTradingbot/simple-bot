"""
Live Market Data Recorder - Minimal Tick Recording for Backtesting
Records only essential data: timestamp, bid, ask, last
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
import gzip

logger = logging.getLogger(__name__)


class LiveDataRecorder:
    """
    Minimal tick-by-tick recorder for realistic backtesting
    Records: timestamp, bid, ask, last - nothing else
    """
    
    def __init__(
        self,
        output_dir: str = "live_data_recordings",
        symbol: str = "ES",
        compress: bool = True,
        max_file_size_mb: int = 100,
        rotation_interval_minutes: int = 60
    ):
        """
        Initialize minimal data recorder
        
        Args:
            output_dir: Directory to save recordings
            symbol: Trading symbol (e.g., "ES", "NQ")
            compress: Use gzip compression (saves space)
            max_file_size_mb: Rotate file when this size is reached
            rotation_interval_minutes: Rotate file every N minutes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.symbol = symbol
        self.compress = compress
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.rotation_interval_seconds = rotation_interval_minutes * 60
        
        # Current file
        self.tick_file = None
        self.tick_file_path = None
        self.tick_file_start_time = None
        
        # Statistics
        self.ticks_recorded = 0
        self.session_start = time.time()
        
        # Initialize file
        self._rotate_tick_file()
        
        logger.info(f"[RECORDER] Initialized - saving to {self.output_dir}")
        logger.info(f"[RECORDER] Recording: {self.tick_file_path.name}")
    
    def _rotate_tick_file(self):
        """Create new tick data file"""
        if self.tick_file:
            self.tick_file.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.symbol}_ticks_{timestamp}.jsonl"
        if self.compress:
            filename += ".gz"
        
        self.tick_file_path = self.output_dir / filename
        
        if self.compress:
            self.tick_file = gzip.open(self.tick_file_path, 'wt', encoding='utf-8')
        else:
            self.tick_file = open(self.tick_file_path, 'w', encoding='utf-8')
        
        self.tick_file_start_time = time.time()
        logger.info(f"[RECORDER] New tick file: {filename}")
    
    def _should_rotate_tick_file(self) -> bool:
        """Check if tick file should be rotated"""
        # Check size
        if self.tick_file_path.exists():
            if self.tick_file_path.stat().st_size > self.max_file_size_bytes:
                return True
        
        # Check time
        if time.time() - self.tick_file_start_time > self.rotation_interval_seconds:
            return True
        
        return False
    
    def record_tick(
        self,
        bid: float,
        ask: float,
        last: float,
        **kwargs  # Ignore all other arguments
    ):
        """
        Record a market tick (only essential data)
        
        Args:
            bid: Best bid price
            ask: Best ask price
            last: Last traded price
        """
        now = time.time()
        
        # Only save essentials: time, bid, ask, last
        tick = {
            "time": now,
            "bid": bid,
            "ask": ask,
            "last": last
        }
        
        # Write to file
        try:
            self.tick_file.write(json.dumps(tick) + '\n')
            self.tick_file.flush()  # Ensure data is written immediately
            self.ticks_recorded += 1
            
            # Rotate if needed
            if self._should_rotate_tick_file():
                self._rotate_tick_file()
                
        except Exception as e:
            logger.error(f"[RECORDER] Failed to record tick: {e}")
    
    def get_stats(self):
        """Get recording statistics"""
        runtime = time.time() - self.session_start
        
        return {
            'session_duration_seconds': round(runtime, 1),
            'ticks_recorded': self.ticks_recorded,
            'ticks_per_second': round(self.ticks_recorded / runtime, 2) if runtime > 0 else 0,
            'current_file': str(self.tick_file_path.name),
            'file_size_mb': round(self.tick_file_path.stat().st_size / (1024*1024), 2) if self.tick_file_path.exists() else 0
        }
    
    def close(self):
        """Close files and finish recording"""
        logger.info("[RECORDER] Closing recorder...")
        
        stats = self.get_stats()
        logger.info(f"[RECORDER] Recorded {stats['ticks_recorded']} ticks in {stats['session_duration_seconds']}s")
        logger.info(f"[RECORDER] File: {stats['current_file']} ({stats['file_size_mb']} MB)")
        
        if self.tick_file:
            self.tick_file.close()
        
        logger.info("[RECORDER] Closed successfully")
