-- PostgreSQL Migration: Create RL Experiences Table
-- Scales to 1000+ concurrent users with instant writes
-- Run this on quotrading-db.postgres.database.azure.com

-- Create table for storing RL trade experiences
CREATE TABLE IF NOT EXISTS rl_experiences (
    id SERIAL PRIMARY KEY,
    license_key VARCHAR(50) NOT NULL,
    symbol VARCHAR(20) NOT NULL,  -- ES, NQ, YM, RTY, etc.
    rsi DECIMAL(5,2) NOT NULL,
    vwap_distance DECIMAL(10,6) NOT NULL,
    atr DECIMAL(10,6) NOT NULL,
    volume_ratio DECIMAL(10,2) NOT NULL,
    hour INTEGER NOT NULL,
    day_of_week INTEGER NOT NULL,
    recent_pnl DECIMAL(10,2) NOT NULL,
    streak INTEGER NOT NULL,
    side VARCHAR(10) NOT NULL,
    regime VARCHAR(50) NOT NULL,
    pnl DECIMAL(10,2) NOT NULL,
    duration DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for fast queries
CREATE INDEX idx_rl_experiences_license ON rl_experiences(license_key);
CREATE INDEX idx_rl_experiences_symbol ON rl_experiences(symbol);
CREATE INDEX idx_rl_experiences_created ON rl_experiences(created_at DESC);
CREATE INDEX idx_rl_experiences_took_trade ON rl_experiences(took_trade);
CREATE INDEX idx_rl_experiences_side ON rl_experiences(side);
CREATE INDEX idx_rl_experiences_regime ON rl_experiences(regime);

-- Composite index for similar state searches (used by RL decision engine)
-- CRITICAL: Symbol is first in index so each symbol's data is isolated
CREATE INDEX idx_rl_experiences_similarity ON rl_experiences(
    symbol, rsi, vwap_distance, atr, volume_ratio, side, regime
);

-- Grant permissions to API user
-- GRANT SELECT, INSERT ON rl_experiences TO quotradingadmin;
-- GRANT USAGE, SELECT ON SEQUENCE rl_experiences_id_seq TO quotradingadmin;

-- Optional: Migrate existing data from Azure Blob to PostgreSQL
-- This is a one-time data migration if you have existing experiences in blob storage
-- You can export blob data and import it here

-- Example stats query (used by submit-outcome endpoint)
-- SELECT 
--     COUNT(*) as total,
--     AVG(CASE WHEN took_trade AND pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
--     AVG(CASE WHEN took_trade THEN pnl ELSE 0 END) as avg_reward
-- FROM rl_experiences
-- WHERE took_trade = TRUE;

-- Example similar states query (used by RL decision engine)
-- SELECT * FROM rl_experiences
-- WHERE side = 'long' AND regime = 'NORMAL'
-- ORDER BY ABS(rsi - 28.5) + ABS(vwap_distance - (-0.45))
-- LIMIT 50;
