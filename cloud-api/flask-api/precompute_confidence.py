"""
Pre-compute confidence buckets for ultra-fast lookups (<100ms)

This script runs nightly (or on-demand) to calculate confidence scores
for all possible market scenarios and store them in Redis.

Result: API requests become instant Redis lookups instead of slow database queries.
"""

import os
import sys
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
import pickle
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Database connection
DB_HOST = os.environ.get('DB_HOST', 'quotrading-db-server.postgres.database.azure.com')
DB_NAME = os.environ.get('DB_NAME', 'quotrading_db')
DB_USER = os.environ.get('DB_USER', 'quotrading_admin')
DB_PASSWORD = os.environ.get('DB_PASSWORD', '')

# Redis connection
REDIS_HOST = os.environ.get('REDIS_HOST', 'quotrading-cache.redis.cache.windows.net')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', '')
REDIS_PORT = int(os.environ.get('REDIS_PORT', '6380'))

# Pre-compute configuration
SYMBOLS = ['ES', 'NQ', 'YM', 'RTY', 'CL', 'GC']  # Add more as needed
REGIMES = [
    'NORMAL', 'NORMAL_TRENDING', 'NORMAL_CHOPPY',
    'HIGH_VOL_CHOPPY', 'HIGH_VOL_TRENDING',
    'LOW_VOL_RANGING', 'LOW_VOL_TRENDING'
]
SIDES = ['LONG', 'SHORT']

# RSI buckets (0-10, 10-20, ..., 90-100)
RSI_BUCKETS = list(range(0, 100, 10))

# VWAP buckets (-2%, -1%, 0%, 1%, 2%)
VWAP_BUCKETS = [-0.02, -0.01, 0.0, 0.01, 0.02]

# ATR buckets (low, medium, high)
ATR_BUCKETS = [1.0, 2.5, 5.0]

# Volume buckets
VOLUME_BUCKETS = [0.5, 1.0, 1.5, 2.0]

# Hour buckets (market open, mid-session, close)
HOUR_BUCKETS = [9, 12, 15]

# Streak buckets
STREAK_BUCKETS = [-3, -1, 0, 1, 3]


def get_bucket(value, buckets):
    """Find the closest bucket for a value"""
    if not buckets:
        return 0
    return min(buckets, key=lambda x: abs(x - value))


def calculate_confidence_for_bucket(experiences, rsi_bucket, vwap_bucket, atr_bucket, 
                                     volume_bucket, hour_bucket, streak_bucket):
    """
    Calculate confidence for a specific bucket of market conditions.
    Uses the same dual pattern matching logic as real-time engine.
    """
    if not experiences:
        return None
    
    # Filter experiences within bucket ranges
    bucket_width = {
        'rsi': 10,
        'vwap': 0.01,
        'atr': 1.5,
        'volume': 0.5,
        'hour': 3,
        'streak': 2
    }
    
    filtered = []
    for exp in experiences:
        if (abs(float(exp['rsi']) - rsi_bucket) <= bucket_width['rsi'] and
            abs(float(exp['vwap_distance']) - vwap_bucket) <= bucket_width['vwap'] and
            abs(float(exp['atr']) - atr_bucket) <= bucket_width['atr'] and
            abs(float(exp['volume_ratio']) - volume_bucket) <= bucket_width['volume'] and
            abs(int(exp['hour']) - hour_bucket) <= bucket_width['hour'] and
            abs(int(exp['streak']) - streak_bucket) <= bucket_width['streak']):
            filtered.append(exp)
    
    if len(filtered) < 5:  # Need minimum experiences for confidence
        return None
    
    # Separate winners and losers
    winners = [e for e in filtered if e['reward'] > 0]
    losers = [e for e in filtered if e['reward'] < 0]
    
    if not winners and not losers:
        return None
    
    # Calculate winner confidence
    winner_confidence = 0.0
    if winners:
        avg_reward = np.mean([w['reward'] for w in winners])
        win_rate = len(winners) / len(filtered)
        winner_confidence = min(1.0, (avg_reward / 100.0) * win_rate)
    
    # Calculate loser penalty
    loser_penalty = 0.0
    if losers:
        avg_loss = abs(np.mean([l['reward'] for l in losers]))
        loss_rate = len(losers) / len(filtered)
        loser_penalty = min(1.0, (avg_loss / 100.0) * loss_rate)
    
    # Final confidence with dual pattern matching
    final_confidence = max(0.0, min(1.0, winner_confidence - (loser_penalty * 0.5)))
    
    return {
        'confidence': round(final_confidence, 4),
        'sample_size': len(filtered),
        'win_rate': round(len(winners) / len(filtered), 4) if filtered else 0,
        'avg_reward': round(np.mean([e['reward'] for e in filtered]), 2) if filtered else 0
    }


def precompute_all_buckets():
    """Pre-compute confidence for all possible market scenarios"""
    
    logging.info("=" * 80)
    logging.info("STARTING PRE-COMPUTE JOB")
    logging.info(f"Time: {datetime.now()}")
    logging.info("=" * 80)
    
    # Connect to database
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode='require'
        )
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        logging.info("✓ Connected to PostgreSQL")
    except Exception as e:
        logging.error(f"✗ Database connection failed: {e}")
        return False
    
    # Connect to Redis
    try:
        redis_client = redis.StrictRedis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            ssl=True,
            ssl_cert_reqs=None,
            decode_responses=False
        )
        redis_client.ping()
        logging.info("✓ Connected to Redis")
    except Exception as e:
        logging.error(f"✗ Redis connection failed: {e}")
        return False
    
    total_buckets = 0
    computed_buckets = 0
    
    # Pre-compute for each symbol/regime/side combination
    for symbol in SYMBOLS:
        for regime in REGIMES:
            for side in SIDES:
                
                # Load all experiences for this combination
                try:
                    cursor.execute("""
                        SELECT rsi, vwap_distance, atr, volume_ratio, hour, 
                               day_of_week, recent_pnl, streak, took_trade, 
                               pnl as reward, duration
                        FROM rl_experiences
                        WHERE symbol = %s AND regime = %s AND side = %s
                        LIMIT 10000
                    """, (symbol, regime, side))
                    
                    experiences = cursor.fetchall()
                    
                    if not experiences:
                        logging.info(f"  Skipping {symbol}/{regime}/{side} (no data)")
                        continue
                    
                    logging.info(f"  Processing {symbol}/{regime}/{side} ({len(experiences)} experiences)")
                    
                except Exception as e:
                    logging.error(f"  Error loading {symbol}/{regime}/{side}: {e}")
                    continue
                
                # Pre-compute for all bucket combinations
                bucket_count = 0
                for rsi_bucket in RSI_BUCKETS:
                    for vwap_bucket in VWAP_BUCKETS:
                        for atr_bucket in ATR_BUCKETS:
                            for volume_bucket in VOLUME_BUCKETS:
                                for hour_bucket in HOUR_BUCKETS:
                                    for streak_bucket in STREAK_BUCKETS:
                                        
                                        total_buckets += 1
                                        
                                        # Calculate confidence for this bucket
                                        result = calculate_confidence_for_bucket(
                                            experiences, rsi_bucket, vwap_bucket, 
                                            atr_bucket, volume_bucket, hour_bucket, 
                                            streak_bucket
                                        )
                                        
                                        if result is None:
                                            continue  # Not enough data for this bucket
                                        
                                        # Store in Redis
                                        cache_key = (
                                            f"confidence_bucket:{symbol}:{regime}:{side}:"
                                            f"{rsi_bucket}:{vwap_bucket}:{atr_bucket}:"
                                            f"{volume_bucket}:{hour_bucket}:{streak_bucket}"
                                        )
                                        
                                        try:
                                            redis_client.setex(
                                                cache_key,
                                                86400,  # 24 hour TTL (refreshes nightly)
                                                pickle.dumps(result)
                                            )
                                            computed_buckets += 1
                                            bucket_count += 1
                                        except Exception as e:
                                            logging.error(f"    Redis write error: {e}")
                
                logging.info(f"    ✓ Computed {bucket_count} buckets")
    
    # Close connections
    cursor.close()
    conn.close()
    redis_client.close()
    
    logging.info("=" * 80)
    logging.info(f"PRE-COMPUTE COMPLETE")
    logging.info(f"Total scenarios evaluated: {total_buckets}")
    logging.info(f"Buckets computed: {computed_buckets}")
    logging.info(f"Coverage: {computed_buckets / total_buckets * 100:.1f}%")
    logging.info(f"Time: {datetime.now()}")
    logging.info("=" * 80)
    
    return True


if __name__ == '__main__':
    success = precompute_all_buckets()
    sys.exit(0 if success else 1)
