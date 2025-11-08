"""
QuoTrading Subscription API - FastAPI Backend
Handles user authentication, license validation, and Stripe subscriptions
"""

from fastapi import FastAPI, HTTPException, Depends, Header, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
import os
import secrets
import hashlib
import hmac
from sqlalchemy import create_engine, Column, String, Boolean, DateTime, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import stripe

# ========================================
# Configuration
# ========================================

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./quotrading.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "change-this-in-production")

stripe.api_key = STRIPE_SECRET_KEY

# ========================================
# Database Setup
# ========================================

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    api_key = Column(String, unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Subscription fields
    subscription_status = Column(String, default="inactive")  # inactive, active, past_due, canceled
    subscription_tier = Column(String, default="basic")  # basic, pro, enterprise
    subscription_start = Column(DateTime, nullable=True)
    subscription_end = Column(DateTime, nullable=True)
    
    # Stripe fields
    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)
    
    # Account limits
    max_contract_size = Column(Integer, default=3)  # Max contracts per trade
    max_accounts = Column(Integer, default=1)  # Number of broker accounts
    
    # Usage tracking
    last_login = Column(DateTime, nullable=True)
    total_logins = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)

# Create tables
Base.metadata.create_all(bind=engine)

# ========================================
# FastAPI App
# ========================================

app = FastAPI(title="QuoTrading API", version="1.0.0")

# CORS - Allow bot to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
# Pydantic Models
# ========================================

class UserCreate(BaseModel):
    email: EmailStr

class LicenseValidation(BaseModel):
    email: EmailStr
    api_key: str

class LicenseResponse(BaseModel):
    valid: bool
    email: str
    subscription_status: str
    subscription_tier: str
    subscription_end: Optional[datetime]
    max_contract_size: int
    max_accounts: int
    message: str

class SubscriptionCreate(BaseModel):
    email: EmailStr
    tier: str  # basic, pro, enterprise
    payment_method_id: str

# ========================================
# Dependencies
# ========================================

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def generate_api_key() -> str:
    """Generate a secure random API key"""
    return f"quo_{secrets.token_urlsafe(32)}"

def hash_email(email: str) -> str:
    """Create deterministic hash of email for lookups"""
    return hashlib.sha256(email.lower().encode()).hexdigest()

# ========================================
# API Endpoints
# ========================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "QuoTrading API",
        "status": "online",
        "version": "1.0.0"
    }

@app.post("/api/v1/users/register")
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    Register a new user and generate API key
    """
    # Check if user exists
    existing_user = db.query(User).filter(User.email == user_data.email.lower()).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    api_key = generate_api_key()
    new_user = User(
        email=user_data.email.lower(),
        api_key=api_key,
        subscription_status="inactive",
        subscription_tier="basic"
    )
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return {
        "success": True,
        "email": new_user.email,
        "api_key": api_key,
        "message": "Registration successful! Check your email for your API key."
    }

@app.post("/api/v1/license/validate", response_model=LicenseResponse)
async def validate_license(credentials: LicenseValidation, db: Session = Depends(get_db)):
    """
    Validate user license and return subscription details
    This is called by the bot on startup
    """
    # Admin master key bypass
    if credentials.api_key == "QUOTRADING_ADMIN_MASTER_2025":
        return LicenseResponse(
            valid=True,
            email=credentials.email,
            subscription_status="active",
            subscription_tier="enterprise",
            subscription_end=datetime.utcnow() + timedelta(days=36500),  # 100 years
            max_contract_size=25,
            max_accounts=999,
            message="Admin access granted"
        )
    
    # Find user by email and API key
    user = db.query(User).filter(
        User.email == credentials.email.lower(),
        User.api_key == credentials.api_key
    ).first()
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or API key"
        )
    
    # Check if subscription is active
    is_valid = False
    message = ""
    
    if user.subscription_status == "active":
        # Check expiration
        if user.subscription_end and user.subscription_end > datetime.utcnow():
            is_valid = True
            message = "License valid - subscription active"
        else:
            message = "Subscription expired - please renew"
    elif user.subscription_status == "past_due":
        message = "Payment past due - please update payment method"
    elif user.subscription_status == "canceled":
        message = "Subscription canceled - please resubscribe"
    else:
        message = "No active subscription - please subscribe to continue"
    
    # Update login stats
    user.last_login = datetime.utcnow()
    user.total_logins += 1
    db.commit()
    
    if not is_valid:
        raise HTTPException(status_code=403, detail=message)
    
    return LicenseResponse(
        valid=is_valid,
        email=user.email,
        subscription_status=user.subscription_status,
        subscription_tier=user.subscription_tier,
        subscription_end=user.subscription_end,
        max_contract_size=user.max_contract_size,
        max_accounts=user.max_accounts,
        message=message
    )

@app.post("/api/v1/subscriptions/create")
async def create_subscription(
    sub_data: SubscriptionCreate,
    db: Session = Depends(get_db)
):
    """
    Create a new Stripe subscription for a user
    """
    # Find user
    user = db.query(User).filter(User.email == sub_data.email.lower()).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Single pricing tier: $200/month
    pricing = {
        "premium": {"monthly": 20000, "contract_limit": 25, "accounts": 10}  # $200/mo
    }
    
    # Force premium tier (only option)
    tier = "premium"
    tier_config = pricing[tier]
    
    try:
        # Create or get Stripe customer
        if not user.stripe_customer_id:
            customer = stripe.Customer.create(
                email=user.email,
                payment_method=sub_data.payment_method_id,
                invoice_settings={"default_payment_method": sub_data.payment_method_id}
            )
            user.stripe_customer_id = customer.id
        
        # Create subscription
        subscription = stripe.Subscription.create(
            customer=user.stripe_customer_id,
            items=[{
                "price_data": {
                    "currency": "usd",
                    "product_data": {
                        "name": f"QuoTrading {sub_data.tier.title()} Plan"
                    },
                    "unit_amount": tier_config["monthly"],
                    "recurring": {"interval": "month"}
                }
            }],
            expand=["latest_invoice.payment_intent"]
        )
        
        # Update user record
        user.stripe_subscription_id = subscription.id
        user.subscription_status = "active"
        user.subscription_tier = "premium"
        user.subscription_start = datetime.utcnow()
        user.subscription_end = datetime.utcnow() + timedelta(days=30)
        user.max_contract_size = tier_config["contract_limit"]
        user.max_accounts = tier_config["accounts"]
        
        db.commit()
        
        return {
            "success": True,
            "subscription_id": subscription.id,
            "status": subscription.status,
            "message": "Successfully subscribed to QuoTrading Premium!"
        }
        
    except stripe.error.CardError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Subscription creation failed: {str(e)}")

@app.post("/api/v1/webhooks/stripe")
async def stripe_webhook(
    request: dict,
    stripe_signature: str = Header(None),
    db: Session = Depends(get_db)
):
    """
    Handle Stripe webhook events (payment success, failure, cancellation)
    """
    if not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")
    
    try:
        # Verify webhook signature
        event = stripe.Webhook.construct_event(
            payload=request,
            sig_header=stripe_signature,
            secret=STRIPE_WEBHOOK_SECRET
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle different event types
    event_type = event["type"]
    
    if event_type == "customer.subscription.updated":
        subscription = event["data"]["object"]
        user = db.query(User).filter(User.stripe_subscription_id == subscription["id"]).first()
        
        if user:
            user.subscription_status = subscription["status"]
            if subscription["status"] == "active":
                user.subscription_end = datetime.fromtimestamp(subscription["current_period_end"])
            db.commit()
    
    elif event_type == "customer.subscription.deleted":
        subscription = event["data"]["object"]
        user = db.query(User).filter(User.stripe_subscription_id == subscription["id"]).first()
        
        if user:
            user.subscription_status = "canceled"
            db.commit()
    
    elif event_type == "invoice.payment_failed":
        invoice = event["data"]["object"]
        subscription_id = invoice.get("subscription")
        user = db.query(User).filter(User.stripe_subscription_id == subscription_id).first()
        
        if user:
            user.subscription_status = "past_due"
            db.commit()
    
    return {"status": "success"}

@app.get("/api/v1/users/{email}")
async def get_user_info(email: str, db: Session = Depends(get_db)):
    """
    Get user subscription information (for dashboard)
    """
    user = db.query(User).filter(User.email == email.lower()).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "email": user.email,
        "subscription_status": user.subscription_status,
        "subscription_tier": user.subscription_tier,
        "subscription_end": user.subscription_end,
        "max_contract_size": user.max_contract_size,
        "max_accounts": user.max_accounts,
        "last_login": user.last_login,
        "total_logins": user.total_logins,
        "created_at": user.created_at
    }

@app.get("/api/v1/admin/dashboard")
async def admin_dashboard(
    admin_key: str = Header(None, alias="X-Admin-Key"),
    db: Session = Depends(get_db)
):
    """
    Admin dashboard - Get all users and stats
    Requires admin key in X-Admin-Key header
    """
    # Verify admin key
    if admin_key != "QUOTRADING_ADMIN_MASTER_2025":
        raise HTTPException(status_code=403, detail="Invalid admin key")
    
    # Get all users
    all_users = db.query(User).all()
    
    # Calculate stats
    total_users = len(all_users)
    active_subscriptions = len([u for u in all_users if u.subscription_status == "active"])
    inactive_users = len([u for u in all_users if u.subscription_status == "inactive"])
    past_due = len([u for u in all_users if u.subscription_status == "past_due"])
    canceled = len([u for u in all_users if u.subscription_status == "canceled"])
    
    # Revenue calculation - Single $200/month tier
    monthly_revenue = active_subscriptions * 200
    
    # User list
    users_list = [
        {
            "email": u.email,
            "status": u.subscription_status,
            "tier": u.subscription_tier,
            "expires": u.subscription_end,
            "last_login": u.last_login,
            "total_logins": u.total_logins,
            "created": u.created_at
        }
        for u in all_users
    ]
    
    return {
        "summary": {
            "total_users": total_users,
            "active_subscriptions": active_subscriptions,
            "inactive_users": inactive_users,
            "past_due": past_due,
            "canceled": canceled,
            "monthly_revenue": f"${monthly_revenue:,.2f}"
        },
        "users": users_list
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
