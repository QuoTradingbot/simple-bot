-- Add device_fingerprint column to users table for session locking
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS device_fingerprint VARCHAR(255);

-- Add device_fingerprint column to heartbeats table for history tracking
ALTER TABLE heartbeats 
ADD COLUMN IF NOT EXISTS device_fingerprint VARCHAR(255);

-- Create index for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_device_fingerprint ON users(device_fingerprint);
CREATE INDEX IF NOT EXISTS idx_users_last_heartbeat ON users(last_heartbeat);
