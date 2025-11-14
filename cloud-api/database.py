"""
Database models and connection management for QuoTrading Cloud API
Supports PostgreSQL for production, SQLite for local development
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime, timedelta
import pytz
import os
from typing import Optional
import secrets

Base = declarative_base()

# Database Models
class User(Base):
    """User account with license information"""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    account_id = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True)
    license_key = Column(String(100), unique=True, nullable=False, index=True)
    license_type = Column(String(50), default='BETA')  # BETA, TRIAL, MONTHLY, ANNUAL
    license_status = Column(String(20), default='ACTIVE')  # ACTIVE, EXPIRED, SUSPENDED
    license_expiration = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(pytz.UTC))
    last_active = Column(DateTime, default=lambda: datetime.now(pytz.UTC))
    is_admin = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    
    # Relationships
    api_logs = relationship("APILog", back_populates="user", cascade="all, delete-orphan")
    trades = relationship("TradeHistory", back_populates="user", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<User {self.account_id} - {self.license_type} ({self.license_status})>"
    
    @property
    def is_license_valid(self) -> bool:
        """Check if license is currently valid"""
        if self.license_status != 'ACTIVE':
            return False
        if self.license_expiration and self.license_expiration < datetime.now(pytz.UTC):
            return False
        return True
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'account_id': self.account_id,
            'email': self.email,
            'license_key': self.license_key,
            'license_type': self.license_type,
            'license_status': self.license_status,
            'license_expiration': self.license_expiration.isoformat() if self.license_expiration else None,
            'is_license_valid': self.is_license_valid,
            'created_at': self.created_at.isoformat(),
            'last_active': self.last_active.isoformat(),
            'is_admin': self.is_admin,
            'notes': self.notes
        }


class APILog(Base):
    """Track API usage per user"""
    __tablename__ = 'api_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Float, nullable=True)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(String(500), nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(pytz.UTC), index=True)
    
    # Relationship
    user = relationship("User", back_populates="api_logs")
    
    def __repr__(self):
        return f"<APILog {self.method} {self.endpoint} - {self.status_code}>"


class TradeHistory(Base):
    """Track user trades for performance analytics"""
    __tablename__ = 'trade_history'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    account_id = Column(String(100), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)  # LONG, SHORT
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    quantity = Column(Integer, nullable=False)
    pnl = Column(Float, nullable=True)
    entry_time = Column(DateTime, default=lambda: datetime.now(pytz.UTC), index=True)
    exit_time = Column(DateTime, nullable=True)
    exit_reason = Column(String(50), nullable=True)
    confidence_score = Column(Float, nullable=True)
    
    # Relationship
    user = relationship("User", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade {self.symbol} {self.signal_type} @ {self.entry_price}>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'account_id': self.account_id,
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'confidence_score': self.confidence_score
        }


class RLExperience(Base):
    """Track RL/ML experiences for signal and exit learning with full context"""
    __tablename__ = 'rl_experiences'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=True, index=True)
    account_id = Column(String(100), nullable=True, index=True)
    experience_type = Column(String(20), nullable=False)  # SIGNAL, EXIT
    symbol = Column(String(20), nullable=True)
    signal_type = Column(String(10), nullable=True)  # LONG, SHORT
    outcome = Column(String(10), nullable=True)  # WIN, LOSS
    pnl = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)  # 0-1, how valuable this experience is
    
    # Context for advanced pattern matching (13 features total)
    # Original 5 features:
    rsi = Column(Float, nullable=True)  # RSI at entry
    vwap_distance = Column(Float, nullable=True)  # Distance from VWAP (percentage)
    vix = Column(Float, nullable=True)  # VIX level at entry
    day_of_week = Column(Integer, nullable=True)  # 0=Monday, 6=Sunday
    hour_of_day = Column(Integer, nullable=True)  # 0-23, UTC
    
    # 8 additional features added for full context:
    atr = Column(Float, nullable=True)  # ATR volatility
    volume_ratio = Column(Float, nullable=True)  # Volume ratio vs average
    recent_pnl = Column(Float, nullable=True)  # Recent P&L (psychological state)
    streak = Column(Integer, nullable=True)  # Win/loss streak
    entry_price = Column(Float, nullable=True)  # Entry price
    vwap = Column(Float, nullable=True)  # VWAP value
    price = Column(Float, nullable=True)  # Current price
    side = Column(String(10), nullable=True)  # long/short
    
    timestamp = Column(DateTime, default=lambda: datetime.now(pytz.UTC), index=True)
    
    # Relationship
    user = relationship("User", backref="rl_experiences")
    
    def __repr__(self):
        return f"<RLExperience {self.experience_type} {self.symbol} - {self.outcome}>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'account_id': self.account_id,
            'experience_type': self.experience_type,
            'symbol': self.symbol,
            'signal_type': self.signal_type,
            'outcome': self.outcome,
            'pnl': self.pnl,
            'confidence_score': self.confidence_score,
            'quality_score': self.quality_score,
            'rsi': self.rsi,
            'vwap_distance': self.vwap_distance,
            'vix': self.vix,
            'day_of_week': self.day_of_week,
            'hour_of_day': self.hour_of_day,
            'timestamp': self.timestamp.isoformat()
        }


class ExitExperience(Base):
    """
    Shared exit learning pool - ALL users contribute and benefit!
    Stores adaptive exit patterns (breakeven, trailing, stop placement, partials)
    """
    __tablename__ = 'exit_experiences'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Market regime when exit occurred
    regime = Column(String(50), nullable=False, index=True)  # HIGH_VOL_TRENDING, NORMAL_CHOPPY, etc.
    
    # Exit parameters used (stored as JSON string)
    exit_params_json = Column(Text, nullable=False)  # {breakeven_threshold_ticks, trailing_distance_ticks, etc.}
    
    # Trade outcome (stored as JSON string)
    outcome_json = Column(Text, nullable=False)  # {pnl, duration, exit_reason, r_multiple, etc.}
    
    # Market situation context (stored as JSON string)
    situation_json = Column(Text, nullable=True)  # {time_of_day, volatility_atr, trend_strength}
    
    # Extended market state (9 features for context-aware learning)
    market_state_json = Column(Text, nullable=True)  # {rsi, volume_ratio, hour, day_of_week, streak, etc.}
    
    # Partial exit decisions (stored as JSON string)
    partial_exits_json = Column(Text, nullable=True)  # [{level, r_multiple, contracts, percentage}, ...]
    
    # Metadata
    timestamp = Column(DateTime, default=lambda: datetime.now(pytz.UTC), index=True)
    quality_score = Column(Float, nullable=True)  # 0-1, how valuable this experience is
    
    def __repr__(self):
        return f"<ExitExperience {self.regime} @ {self.timestamp}>"
    
    def to_dict(self):
        """Convert to dictionary for API responses (matches JSON file format)"""
        import json
        return {
            'id': self.id,
            'timestamp': self.timestamp.isoformat(),
            'regime': self.regime,
            'exit_params': json.loads(self.exit_params_json),
            'outcome': json.loads(self.outcome_json),
            'situation': json.loads(self.situation_json) if self.situation_json else {},
            'market_state': json.loads(self.market_state_json) if self.market_state_json else {},
            'partial_exits': json.loads(self.partial_exits_json) if self.partial_exits_json else [],
            'quality_score': self.quality_score
        }


class MLExperience(Base):
    """
    NEW: Complete ML experience storage with JSONB for flexible RL state + outcome.
    Supports signal experiences, exit experiences, AND ghost trades (shadow tracking).
    Multi-user learning pool - all experiences aggregated for daily model retraining.
    """
    __tablename__ = 'ml_experiences'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), nullable=False, index=True)  # Privacy-preserving hash
    symbol = Column(String(20), nullable=False, index=True)  # ES, NQ, MES, etc.
    experience_type = Column(String(20), nullable=False, index=True)  # 'signal', 'exit', 'ghost_trade'
    
    # Complete RL state (26-45 features depending on experience type) - JSONB for fast queries
    rl_state = Column(JSONB, nullable=False)  # {rsi, vwap_distance, vix, atr, volume_ratio, ...}
    
    # Complete outcome data - JSONB for flexible schema
    outcome = Column(JSONB, nullable=False)  # {pnl, exit_reason, mae, mfe, breakeven_activated, ...}
    
    # Metadata
    timestamp = Column(DateTime, default=lambda: datetime.now(pytz.UTC), index=True)
    quality_score = Column(Float, nullable=True)  # 0-1, calculated quality of this experience
    
    # Indexes for fast queries
    # CREATE INDEX idx_ml_exp_type_time ON ml_experiences(experience_type, timestamp DESC);
    # CREATE INDEX idx_ml_exp_user_symbol ON ml_experiences(user_id, symbol);
    # CREATE INDEX idx_ml_exp_rl_state ON ml_experiences USING GIN(rl_state);
    # CREATE INDEX idx_ml_exp_outcome ON ml_experiences USING GIN(outcome);
    
    def __repr__(self):
        return f"<MLExperience {self.experience_type} {self.symbol} @ {self.timestamp}>"
    
    def to_dict(self):
        """Convert to dictionary for training scripts"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'symbol': self.symbol,
            'experience_type': self.experience_type,
            'rl_state': self.rl_state,  # Already dict (JSONB)
            'outcome': self.outcome,  # Already dict (JSONB)
            'timestamp': self.timestamp.isoformat(),
            'quality_score': self.quality_score
        }


class ModelVersion(Base):
    """
    NEW: Model version tracking with Azure Blob Storage URLs.
    Tracks signal_model.pth and exit_model.pth versions + performance metrics.
    """
    __tablename__ = 'model_versions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(20), nullable=False, index=True)  # 'signal' or 'exit'
    version = Column(String(50), nullable=False, unique=True, index=True)  # e.g., 'v1.2.3' or '2025-11-14_001'
    
    # Model file location
    blob_url = Column(String(500), nullable=False)  # Azure Blob Storage URL with SAS token
    blob_container = Column(String(100), nullable=False, default='models')  # Container name
    blob_path = Column(String(255), nullable=False)  # Path within container: signal_model_v1.2.3.pth
    
    # Training metadata
    training_start = Column(DateTime, nullable=False)
    training_end = Column(DateTime, nullable=False)
    training_duration_mins = Column(Integer, nullable=True)
    
    # Performance metrics (stored as JSONB for flexibility)
    metrics = Column(JSONB, nullable=False)  # {accuracy, loss, val_loss, win_rate, sharpe, etc.}
    
    # Training data info
    num_experiences = Column(Integer, nullable=False)  # How many experiences used for training
    num_users = Column(Integer, nullable=True)  # How many users contributed data
    
    # Deployment status
    is_deployed = Column(Boolean, default=False, index=True)  # Currently active version?
    deployed_at = Column(DateTime, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(pytz.UTC), index=True)
    created_by = Column(String(100), nullable=True)  # Who/what triggered training
    notes = Column(Text, nullable=True)  # Training notes, issues, insights
    
    def __repr__(self):
        return f"<ModelVersion {self.model_type} {self.version} {'[DEPLOYED]' if self.is_deployed else ''}>"
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'model_type': self.model_type,
            'version': self.version,
            'blob_url': self.blob_url,
            'blob_path': self.blob_path,
            'metrics': self.metrics,  # Already dict (JSONB)
            'num_experiences': self.num_experiences,
            'num_users': self.num_users,
            'is_deployed': self.is_deployed,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None,
            'created_at': self.created_at.isoformat(),
            'training_duration_mins': self.training_duration_mins
        }


# Database Connection Manager
class DatabaseManager:
    """Manages database connections and sessions"""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            database_url: PostgreSQL URL or SQLite path
                         Default: Uses DATABASE_URL env var or SQLite
        """
        if database_url is None:
            database_url = os.getenv('DATABASE_URL', 'sqlite:///./quotrading.db')
        
        # Handle Azure PostgreSQL connection strings
        if database_url.startswith('postgresql://'):
            # Azure requires SSL
            if 'sslmode=' not in database_url:
                database_url += '?sslmode=require'
        
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections after 1 hour
            echo=False           # Set to True for SQL debugging
        )
        
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.database_url = database_url
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        print("✅ Database tables created successfully")
    
    def drop_tables(self):
        """Drop all database tables (use with caution!)"""
        Base.metadata.drop_all(bind=self.engine)
        print("⚠️  All database tables dropped")
    
    def get_session(self):
        """Get a new database session"""
        return self.SessionLocal()
    
    def get_engine(self):
        """Get the SQLAlchemy engine"""
        return self.engine
    
    def close(self):
        """Close database connection"""
        self.engine.dispose()


# Helper Functions
def generate_license_key(prefix: str = "QT") -> str:
    """Generate a secure license key"""
    random_part = secrets.token_hex(12).upper()
    return f"{prefix}-{random_part[:4]}-{random_part[4:8]}-{random_part[8:12]}-{random_part[12:16]}"


def create_user(
    db_session,
    account_id: str,
    email: Optional[str] = None,
    license_type: str = 'BETA',
    license_duration_days: Optional[int] = None,
    is_admin: bool = False,
    notes: Optional[str] = None
) -> User:
    """
    Create a new user with license
    
    Args:
        db_session: SQLAlchemy session
        account_id: Unique account identifier
        email: User email (optional)
        license_type: BETA, TRIAL, MONTHLY, ANNUAL
        license_duration_days: License duration (None = unlimited for BETA)
        is_admin: Admin privileges
        notes: Additional notes
    
    Returns:
        User object
    """
    # Generate unique license key
    license_key = generate_license_key()
    
    # Set expiration
    expiration = None
    if license_duration_days:
        expiration = datetime.now(pytz.UTC) + timedelta(days=license_duration_days)
    
    user = User(
        account_id=account_id,
        email=email,
        license_key=license_key,
        license_type=license_type,
        license_status='ACTIVE',
        license_expiration=expiration,
        is_admin=is_admin,
        notes=notes
    )
    
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    
    return user


def get_user_by_license_key(db_session, license_key: str) -> Optional[User]:
    """Get user by license key"""
    return db_session.query(User).filter(User.license_key == license_key).first()


def get_user_by_account_id(db_session, account_id: str) -> Optional[User]:
    """Get user by account ID"""
    return db_session.query(User).filter(User.account_id == account_id).first()


def log_api_call(
    db_session,
    user_id: int,
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: Optional[float] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None
):
    """Log an API call"""
    log = APILog(
        user_id=user_id,
        endpoint=endpoint,
        method=method,
        status_code=status_code,
        response_time_ms=response_time_ms,
        ip_address=ip_address,
        user_agent=user_agent
    )
    db_session.add(log)
    db_session.commit()


def update_user_activity(db_session, user_id: int):
    """Update user's last active timestamp"""
    user = db_session.query(User).filter(User.id == user_id).first()
    if user:
        user.last_active = datetime.now(pytz.UTC)
        db_session.commit()


# Initialize global database manager (will be set in main.py)
db_manager: Optional[DatabaseManager] = None


def get_db():
    """Dependency for FastAPI endpoints to get database session"""
    if db_manager is None:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    db = db_manager.get_session()
    try:
        yield db
    finally:
        db.close()
