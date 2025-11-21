"""
Database models and connection management for QuoTrading Cloud API
Supports PostgreSQL for production, SQLite for local development
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timedelta
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
    created_at = Column(DateTime, default=datetime.utcnow)
    last_active = Column(DateTime, default=datetime.utcnow)
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
        if self.license_expiration and self.license_expiration < datetime.utcnow():
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
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
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
    entry_time = Column(DateTime, default=datetime.utcnow, index=True)
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
    """Track RL/ML experiences for signal learning with full context"""
    __tablename__ = 'rl_experiences'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    account_id = Column(String(100), nullable=False, index=True)
    experience_type = Column(String(20), nullable=False)  # SIGNAL (future: other types)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)  # LONG, SHORT
    outcome = Column(String(10), nullable=False)  # WIN, LOSS
    pnl = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)  # 0-1, how valuable this experience is
    
    # NEW: Context for advanced pattern matching
    rsi = Column(Float, nullable=True)  # RSI at entry
    vwap_distance = Column(Float, nullable=True)  # Distance from VWAP (percentage)
    vix = Column(Float, nullable=True)  # VIX level at entry
    day_of_week = Column(Integer, nullable=True)  # 0=Monday, 6=Sunday
    hour_of_day = Column(Integer, nullable=True)  # 0-23, Eastern Time
    
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
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
        expiration = datetime.utcnow() + timedelta(days=license_duration_days)
    
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
        user.last_active = datetime.utcnow()
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
