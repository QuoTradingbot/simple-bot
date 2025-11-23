"""
QuoTrading Flask API with RL Brain + PostgreSQL License Validation
Simple, reliable API that works everywhere
"""
from flask import Flask, request, jsonify
import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import logging
import statistics
import secrets
import string
import stripe
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from rl_decision_engine import CloudRLDecisionEngine

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# PostgreSQL configuration
DB_HOST = os.environ.get("DB_HOST", "quotrading-db.postgres.database.azure.com")
DB_NAME = os.environ.get("DB_NAME", "quotrading")
DB_USER = os.environ.get("DB_USER", "quotradingadmin")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_PORT = os.environ.get("DB_PORT", "5432")

# Stripe configuration
stripe.api_key = os.environ.get("STRIPE_API_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
ADMIN_API_KEY = os.environ.get("ADMIN_API_KEY", "ADMIN-DEV-KEY-2024")  # For creating licenses

# Email configuration (for SendGrid or SMTP)
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "noreply@quotrading.com")

# Download link for the bot EXE
BOT_DOWNLOAD_URL = os.environ.get("BOT_DOWNLOAD_URL", "https://your-download-link.com/QuoTrading_Bot.exe")

# Global cache with time-based expiration (30 seconds)
_experiences_cache = None
_experiences_cache_time = None
_CACHE_EXPIRATION_SECONDS = 30

def send_license_email(email, license_key):
    """Send email with license key and download link"""
    try:
        subject = "Your QuoTrading AI License Key & Download Link"
        
        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h1 style="color: #667eea;">Welcome to QuoTrading AI! 🎉</h1>
            
            <p>Thank you for your purchase! Here's everything you need to get started:</p>
            
            <div style="background: #f3f4f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h2 style="margin-top: 0;">Your License Key:</h2>
                <p style="font-size: 24px; font-weight: bold; color: #667eea; letter-spacing: 2px;">{license_key}</p>
            </div>
            
            <h2>Getting Started:</h2>
            <ol>
                <li><strong>Download the Bot:</strong> <a href="{BOT_DOWNLOAD_URL}" style="color: #667eea;">Click here to download</a></li>
                <li><strong>Run the EXE file</strong> on your Windows computer</li>
                <li><strong>Enter your license key</strong> when prompted: <code>{license_key}</code></li>
                <li><strong>Configure your broker</strong> settings in the bot</li>
                <li><strong>Start trading!</strong> The AI will begin analyzing markets</li>
            </ol>
            
            <h2>Need Help?</h2>
            <p>Join our Discord community or email support@quotrading.com</p>
            
            <p style="color: #6b7280; font-size: 12px; margin-top: 40px;">
                Your subscription will auto-renew monthly. You can manage your subscription anytime from your Stripe customer portal.
            </p>
        </body>
        </html>
        """
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = FROM_EMAIL
        msg['To'] = email
        msg.attach(MIMEText(html_body, 'html'))
        
        # Send via SMTP
        if SMTP_USERNAME and SMTP_PASSWORD:
            with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
            logging.info(f"📧 Email sent to {email}")
            return True
        else:
            logging.warning(f"⚠️ Email not configured - would send license {license_key} to {email}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Failed to send email: {e}")
        return False

def generate_license_key():
    """Generate a unique license key"""
    characters = string.ascii_uppercase + string.digits
    segments = []
    for _ in range(4):
        segment = ''.join(secrets.choice(characters) for _ in range(4))
        segments.append(segment)
    return '-'.join(segments)  # Format: XXXX-XXXX-XXXX-XXXX

def get_db_connection():
    """Get PostgreSQL database connection"""
    try:
        # Try standard username first, then flexible server format if needed
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD,
                sslmode='require',
                connect_timeout=10
            )
            return conn
        except psycopg2.OperationalError:
            # Fallback for flexible server if simple username fails
            user_with_server = f"{DB_USER}@{DB_HOST.split('.')[0]}" if '@' not in DB_USER else DB_USER
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=user_with_server,
                password=DB_PASSWORD,
                sslmode='require',
                connect_timeout=10
            )
            return conn
            
    except Exception as e:
        logging.error(f"❌ Database connection failed: {e}")
        logging.error(f"   Host: {DB_HOST}, User: {DB_USER}, DB: {DB_NAME}")
        return None

def validate_license(license_key: str):
    """Validate license key against PostgreSQL database
    
    Returns:
        Tuple of (is_valid: bool, message: str, expiration_date: datetime or None)
    """
    if not license_key:
        return False, "License key required", None
    
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed", None
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT license_key, email, license_type, license_status, 
                       license_expiration, created_at
                FROM users 
                WHERE license_key = %s
            """, (license_key,))
            
            user = cursor.fetchone()
            
            if not user:
                return False, "Invalid license key", None
            
            # Check if license is active (case-insensitive)
            if user['license_status'].lower() != 'active':
                return False, f"License is {user['license_status']}", user['license_expiration']
            
            # Check expiration
            if user['license_expiration']:
                if datetime.now() > user['license_expiration']:
                    return False, "License expired", user['license_expiration']
            
            # Log successful validation
            cursor.execute("""
                INSERT INTO api_logs (license_key, endpoint, request_data, status_code)
                VALUES (%s, %s, %s, %s)
            """, (license_key, '/api/main', '{"action": "validate"}', 200))
            conn.commit()
            
            return True, f"Valid {user['license_type']} license", user['license_expiration']
            
    except Exception as e:
        logging.error(f"License validation error: {e}")
        return False, str(e), None
    finally:
        conn.close()

def load_experiences(symbol='ES'):
    """
    Load RL experiences from PostgreSQL database for specific symbol.
    Each symbol (ES, NQ, YM, etc.) has separate RL brain.
    Uses PostgreSQL for scalability and performance with 1000+ concurrent users.
    Implements 30-second cache for optimal performance.
    
    Args:
        symbol: Trading symbol (ES, NQ, YM, RTY, etc.)
    """
    global _experiences_cache, _experiences_cache_time
    
    # Cache key includes symbol to separate ES vs NQ vs YM brains
    cache_key = f"{symbol}_experiences"
    
    # Check if cache is still valid for this symbol
    if (_experiences_cache is not None and 
        _experiences_cache_time is not None and
        _experiences_cache.get('symbol') == symbol):
        cache_age = (datetime.now() - _experiences_cache_time).total_seconds()
        if cache_age < _CACHE_EXPIRATION_SECONDS:
            return _experiences_cache.get('data', [])
        else:
            logging.info(f"Cache expired ({cache_age:.1f}s old) - reloading {symbol} from database")
    
    try:
        conn = get_db_connection()
        if not conn:
            logging.error("Database unavailable - cannot load experiences")
            return None  # Return None to signal connection failure
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        # Load all experiences from database for this symbol
        cursor.execute("""
            SELECT 
                rsi,
                vwap_distance,
                atr,
                volume_ratio,
                hour,
                day_of_week,
                recent_pnl,
                streak,
                side,
                regime,
                took_trade,
                pnl,
                duration
            FROM rl_experiences
            WHERE symbol = %s
            ORDER BY created_at DESC
            LIMIT 10000
        """, (symbol,))
        
        rows = cursor.fetchall()
        
        # Convert to RL engine format (nested structure)
        experiences = []
        for row in rows:
            experiences.append({
                'state': {
                    'rsi': float(row['rsi']),
                    'vwap_distance': float(row['vwap_distance']),
                    'atr': float(row['atr']),
                    'volume_ratio': float(row['volume_ratio']),
                    'hour': int(row['hour']),
                    'day_of_week': int(row['day_of_week']),
                    'recent_pnl': float(row['recent_pnl']),
                    'streak': int(row['streak']),
                    'side': str(row['side']),
                    'regime': str(row['regime'])
                },
                'action': {
                    'took_trade': bool(row['took_trade'])
                },
                'reward': float(row['pnl']),  # Map pnl to reward
                'duration': float(row['duration'])
            })
        
        cursor.close()
        conn.close()
        
        # Cache with symbol identifier
        _experiences_cache = {
            'symbol': symbol,
            'data': experiences
        }
        _experiences_cache_time = datetime.now()
        logging.info(f"✅ Loaded {len(experiences)} {symbol} experiences from PostgreSQL")
        
        return experiences
        
    except Exception as e:
        logging.error(f"❌ Failed to load experiences from database: {e}")
        _experiences_cache = []
        _experiences_cache_time = datetime.now()
        return _experiences_cache

def calculate_confidence(signal_type: str, regime: str, vix_level: float, experiences: list) -> float:
    """Calculate signal confidence based on RL experiences"""
    if not experiences:
        return 0.5
    
    # Filter similar experiences
    similar = [
        exp for exp in experiences
        if exp.get('signal_type') == signal_type and exp.get('regime') == regime
    ]
    
    if not similar:
        return 0.5
    
    # Calculate average reward from similar experiences
    rewards = [exp.get('reward', 0) for exp in similar]
    avg_reward = statistics.mean(rewards) if rewards else 0
    
    # Convert reward to confidence (0.0 to 1.0)
    confidence = max(0.0, min(1.0, (avg_reward + 1.0) / 2.0))
    
    return round(confidence, 3)

@app.route('/api/hello', methods=['GET'])
def hello():
    """Health check endpoint"""
    experiences = load_experiences()
    return jsonify({
        "status": "success",
        "message": f"✅ QuoTrading Flask API is running!",
        "experiences_loaded": len(experiences),
        "database_configured": bool(DB_PASSWORD)
    }), 200

@app.route('/api/main', methods=['POST'])
def main():
    """Main signal processing endpoint with license validation"""
    try:
        data = request.get_json()
        
        # Validate license
        license_key = data.get('license_key')
        is_valid, message, expiration_date = validate_license(license_key)
        
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": message,
                "license_valid": False,
                "license_expiration": expiration_date.isoformat() if expiration_date else None
            }), 403
        
        # Process signal with RL brain
        signal_type = data.get('signal_type', 'NEUTRAL')
        regime = data.get('regime', 'RANGING')
        vix_level = data.get('vix_level', 15.0)
        
        experiences = load_experiences()
        confidence = calculate_confidence(signal_type, regime, vix_level, experiences)
        
        # Calculate days until expiration
        days_until_expiration = None
        hours_until_expiration = None
        if expiration_date:
            time_until_expiration = expiration_date - datetime.now()
            days_until_expiration = time_until_expiration.days
            hours_until_expiration = time_until_expiration.total_seconds() / 3600
        
        response = {
            "status": "success",
            "license_valid": True,
            "message": message,
            "license_expiration": expiration_date.isoformat() if expiration_date else None,
            "days_until_expiration": days_until_expiration,
            "hours_until_expiration": hours_until_expiration,
            "signal_confidence": confidence,
            "experiences_used": len(experiences),
            "signal_type": signal_type,
            "regime": regime
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/stripe/create-checkout', methods=['POST'])
def create_checkout_session():
    """Create a Stripe checkout session for subscription purchase"""
    try:
        data = request.get_json()
        email = data.get('email')
        plan = data.get('plan', 'monthly')  # monthly or yearly
        
        if not email:
            return jsonify({"status": "error", "message": "Email required"}), 400
        
        # Check if Stripe is configured
        if not stripe.api_key or stripe.api_key == "":
            return jsonify({
                "status": "error",
                "message": "Stripe not configured. Set STRIPE_SECRET_KEY environment variable."
            }), 500
        
        # Determine price based on plan (using your Stripe price)
        # QuoTrading Bot - $200/month subscription
        prices = {
            'monthly': 'price_1SWBk4P0y2Nhiub4js7bhUD7',  # QuoTrading Bot - $200/month
            'yearly': 'price_1SWBk4P0y2Nhiub4js7bhUD7'    # Use same for now
        }
        
        price_id = prices.get(plan, 'price_1SWBk4P0y2Nhiub4js7bhUD7')  # Default to monthly
        
        # Create Stripe checkout session
        session = stripe.checkout.Session.create(
            customer_email=email,
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url='https://quotrading-flask-api.azurewebsites.net/api/stripe/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url='https://quotrading-flask-api.azurewebsites.net/api/stripe/cancel',
            allow_promotion_codes=True,  # Allow discount codes
            billing_address_collection='auto',
            metadata={
                'plan': plan,
                'email': email
            }
        )
        
        return jsonify({
            "status": "success",
            "checkout_url": session.url,
            "session_id": session.id
        }), 200
        
    except stripe.error.StripeError as e:
        logging.error(f"Stripe error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    except Exception as e:
        logging.error(f"Error creating checkout session: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stripe/customer-portal', methods=['POST'])
def create_customer_portal():
    """Create a Stripe customer portal session for subscription management"""
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({"status": "error", "message": "Email required"}), 400
        
        # Find customer by email
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database error"}), 500
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT stripe_customer_id FROM users WHERE email = %s AND license_status = 'active'
                """, (email,))
                result = cursor.fetchone()
                
                if not result or not result[0]:
                    return jsonify({"status": "error", "message": "No active subscription found"}), 404
                
                customer_id = result[0]
                
                # Create customer portal session
                session = stripe.billing_portal.Session.create(
                    customer=customer_id,
                    return_url='https://quotrading-flask-api.azurewebsites.net/api/stripe/success',
                )
                
                return jsonify({
                    "status": "success",
                    "portal_url": session.url
                }), 200
                
        finally:
            conn.close()
            
    except Exception as e:
        logging.error(f"Error creating customer portal: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/list-licenses', methods=['GET'])
def list_licenses():
    """List all licenses with details (admin only)"""
    try:
        # Verify admin API key
        api_key = request.headers.get('X-Admin-Key') or request.args.get('admin_key')
        if api_key != ADMIN_API_KEY:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database error"}), 500
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        license_key, 
                        email, 
                        license_type, 
                        license_status, 
                        license_expiration,
                        created_at
                    FROM users 
                    ORDER BY created_at DESC
                """)
                licenses = cursor.fetchall()
                
                license_list = []
                for lic in licenses:
                    license_list.append({
                        "license_key": lic[0],
                        "email": lic[1],
                        "type": lic[2],
                        "status": lic[3],
                        "expires_at": lic[4].isoformat() if lic[4] else None,
                        "created_at": lic[5].isoformat() if lic[5] else None
                    })
                
                return jsonify({
                    "status": "success",
                    "total_licenses": len(license_list),
                    "licenses": license_list
                }), 200
                
        finally:
            conn.close()
            
    except Exception as e:
        logging.error(f"Error listing licenses: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/update-license-status', methods=['POST'])
def update_license_status():
    """Update license status - ban/suspend/activate (admin only)"""
    try:
        data = request.get_json()
        
        # Verify admin API key
        api_key = request.headers.get('X-Admin-Key') or data.get('admin_key')
        if api_key != ADMIN_API_KEY:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
        license_key = data.get('license_key')
        new_status = data.get('status')  # 'active', 'suspended', 'expired', 'cancelled'
        
        if not license_key or not new_status:
            return jsonify({"status": "error", "message": "license_key and status required"}), 400
        
        if new_status not in ['active', 'suspended', 'expired', 'cancelled']:
            return jsonify({"status": "error", "message": "Invalid status"}), 400
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database error"}), 500
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users 
                    SET license_status = %s 
                    WHERE license_key = %s
                    RETURNING email, license_type
                """, (new_status, license_key))
                result = cursor.fetchone()
                conn.commit()
                
                if not result:
                    return jsonify({"status": "error", "message": "License not found"}), 404
                
                return jsonify({
                    "status": "success",
                    "message": f"License {license_key} status updated to {new_status}",
                    "email": result[0],
                    "license_type": result[1],
                    "new_status": new_status
                }), 200
                
        finally:
            conn.close()
            
    except Exception as e:
        logging.error(f"Error updating license status: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/admin/create-license', methods=['POST'])
def create_license():
    """Create a new license (admin only)"""
    try:
        data = request.get_json()
        
        # Verify admin API key
        api_key = request.headers.get('X-Admin-Key') or data.get('admin_key')
        if api_key != ADMIN_API_KEY:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
        email = data.get('email')
        license_type = data.get('license_type', 'standard')
        duration_days = data.get('duration_days', 30)
        
        if not email:
            return jsonify({"status": "error", "message": "Email required"}), 400
        
        license_key = generate_license_key()
        expiration = datetime.now() + timedelta(days=duration_days)
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database connection failed"}), 500
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO users (license_key, email, license_type, license_status, license_expiration)
                    VALUES (%s, %s, %s, %s, %s)
                """, (license_key, email, license_type, 'active', expiration))
                conn.commit()
                
            return jsonify({
                "status": "success",
                "license_key": license_key,
                "email": email,
                "license_type": license_type,
                "expires_at": expiration.isoformat(),
                "duration_days": duration_days
            }), 201
            
        finally:
            conn.close()
            
    except Exception as e:
        logging.error(f"Error creating license: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/stripe/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events for subscription management"""
    payload = request.data
    sig_header = request.headers.get('Stripe-Signature')
    
    if not STRIPE_WEBHOOK_SECRET:
        logging.warning("⚠️ Stripe webhook secret not configured - using raw payload")
        event = json.loads(payload)
    else:
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        except ValueError:
            logging.error("Invalid payload")
            return jsonify({"status": "error", "message": "Invalid payload"}), 400
        except stripe.error.SignatureVerificationError:
            logging.error("Invalid signature")
            return jsonify({"status": "error", "message": "Invalid signature"}), 400
    
    event_type = event['type']
    data = event['data']['object']
    
    logging.info(f"📬 Stripe webhook: {event_type}")
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"status": "error", "message": "Database error"}), 500
    
    try:
        with conn.cursor() as cursor:
            if event_type == 'checkout.session.completed':
                # Payment successful - create license
                customer_email = data.get('customer_email')
                subscription_id = data.get('subscription')
                
                if customer_email:
                    license_key = generate_license_key()
                    
                    # Subscription licenses don't expire (auto-renew monthly)
                    cursor.execute("""
                        INSERT INTO users (license_key, email, license_type, license_status, license_expiration)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (license_key, customer_email, 'subscription', 'active', None))
                    conn.commit()
                    
                    logging.info(f"🎉 License created from Stripe payment: {license_key} for {customer_email}")
                    
                    # Send email with license key and download link
                    send_license_email(customer_email, license_key)
            
            elif event_type == 'customer.subscription.deleted':
                # Subscription cancelled - revoke license
                customer_id = data.get('customer')
                
                # Find and cancel all licenses for this customer
                cursor.execute("""
                    UPDATE users 
                    SET license_status = 'cancelled', license_expiration = NOW()
                    WHERE email = (SELECT email FROM users WHERE license_key LIKE %s LIMIT 1)
                    AND license_status = 'active'
                """, (f"%{customer_id}%",))
                conn.commit()
                
                logging.info(f"❌ Licenses cancelled for customer (subscription ended)")
            
            elif event_type == 'invoice.payment_failed':
                # Payment failed - suspend license
                customer_email = data.get('customer_email')
                
                if customer_email:
                    cursor.execute("""
                        UPDATE users 
                        SET license_status = 'suspended'
                        WHERE email = %s AND license_status = 'active'
                    """, (customer_email,))
                    conn.commit()
                    
                    logging.warning(f"⚠️ License suspended for {customer_email} (payment failed)")
        
        return jsonify({"status": "success"}), 200
        
    except Exception as e:
        logging.error(f"Webhook processing error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        conn.close()

# ============================================================================
# ADMIN DASHBOARD ENDPOINTS
# ============================================================================

@app.route('/api/admin/dashboard-stats', methods=['GET'])
def admin_dashboard_stats():
    """Get overall dashboard statistics"""
    admin_key = request.args.get('license_key') or request.args.get('admin_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Total users
            cursor.execute("SELECT COUNT(*) as total FROM users")
            total_users = cursor.fetchone()['total']
            
            # Active licenses
            cursor.execute("SELECT COUNT(*) as active FROM users WHERE license_status = 'ACTIVE'")
            active_licenses = cursor.fetchone()['active']
            
            # Online users (active in last 5 minutes based on API logs)
            cursor.execute("""
                SELECT COUNT(DISTINCT license_key) as online FROM api_logs
                WHERE created_at > NOW() - INTERVAL '5 minutes'
            """)
            online_users = cursor.fetchone()['online']
            
            # API calls in last 24 hours
            cursor.execute("""
                SELECT COUNT(*) as count FROM api_logs
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """)
            api_calls_24h = cursor.fetchone()['count']
            
            # Get RL experience counts
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE created_at > NOW() - INTERVAL '24 hours') as today
                FROM rl_experiences
                WHERE took_trade = TRUE
            """)
            rl_stats = cursor.fetchone()
            signal_exp_total = rl_stats['total'] or 0
            signal_exp_24h = rl_stats['today'] or 0
            
            # Get trade statistics (total trades and P&L)
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl
                FROM rl_experiences
                WHERE took_trade = TRUE
            """)
            trade_stats = cursor.fetchone()
            total_trades = trade_stats['total_trades'] or 0
            total_pnl = float(trade_stats['total_pnl']) if trade_stats['total_pnl'] else 0.0
            
            # Calculate revenue metrics
            pricing = {
                'MONTHLY': 200.00,
                'ANNUAL': 2000.00,
                'TRIAL': 0.00,
                'BETA': 0.00
            }
            
            # Get active subscriptions breakdown
            cursor.execute("""
                SELECT COUNT(*) as count, UPPER(license_type) as type
                FROM users
                WHERE UPPER(license_status) = 'ACTIVE'
                GROUP BY UPPER(license_type)
            """)
            active_breakdown = cursor.fetchall()
            
            # Calculate MRR (Monthly Recurring Revenue)
            mrr = sum(
                r['count'] * (pricing.get(r['type'], 0) if r['type'] == 'MONTHLY' 
                             else pricing.get(r['type'], 0) / 12) 
                for r in active_breakdown
            )
            
            # Calculate ARR (Annual Recurring Revenue)
            arr = mrr * 12
            
            return jsonify({
                "users": {
                    "total": total_users,
                    "active": active_licenses,
                    "online_now": online_users
                },
                "api_calls": {
                    "last_24h": api_calls_24h
                },
                "trades": {
                    "total": total_trades,
                    "total_pnl": total_pnl
                },
                "rl_experiences": {
                    "total_signal_experiences": signal_exp_total,
                    "signal_experiences_24h": signal_exp_24h
                },
                "revenue": {
                    "mrr": round(mrr, 2),
                    "arr": round(arr, 2),
                    "active_subscriptions": active_breakdown
                }
            }), 200
    except Exception as e:
        logging.error(f"Dashboard stats error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/admin/users', methods=['GET'])
def admin_list_users():
    """List all users (same as list-licenses but formatted for dashboard)"""
    admin_key = request.args.get('license_key') or request.args.get('admin_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT u.id, u.email, u.license_key, u.license_type, u.license_status, 
                       u.license_expiration, u.created_at,
                       MAX(a.created_at) as last_active,
                       CASE WHEN MAX(a.created_at) > NOW() - INTERVAL '5 minutes' 
                            THEN true ELSE false END as is_online,
                       COUNT(a.id) as api_call_count,
                       (SELECT COUNT(*) FROM rl_experiences r 
                        WHERE r.license_key = u.license_key AND r.took_trade = TRUE) as trade_count
                FROM users u
                LEFT JOIN api_logs a ON u.license_key = a.license_key
                GROUP BY u.id, u.email, u.license_key, u.license_type, u.license_status, 
                         u.license_expiration, u.created_at
                ORDER BY u.created_at DESC
            """)
            users = cursor.fetchall()
            
            # Format for dashboard (use account_id instead of id for compatibility)
            formatted_users = []
            for user in users:
                formatted_users.append({
                    "account_id": str(user['id']),
                    "email": user['email'],
                    "license_key": user['license_key'],
                    "license_type": user['license_type'].upper() if user['license_type'] else 'MONTHLY',
                    "license_status": user['license_status'].upper() if user['license_status'] else 'ACTIVE',
                    "license_expiration": user['license_expiration'].isoformat() if user['license_expiration'] else None,
                    "created_at": user['created_at'].isoformat() if user['created_at'] else None,
                    "last_active": user['last_active'].isoformat() if user['last_active'] else None,
                    "is_online": user['is_online'],
                    "api_call_count": int(user['api_call_count']) if user['api_call_count'] else 0,
                    "trade_count": int(user['trade_count']) if user['trade_count'] else 0
                })
            
            return jsonify({"users": formatted_users}), 200
    except Exception as e:
        logging.error(f"List users error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/admin/user/<account_id>', methods=['GET'])
def admin_get_user(account_id):
    """Get detailed information about a specific user"""
    admin_key = request.args.get('license_key') or request.args.get('admin_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get user details with last active from api_logs
            cursor.execute("""
                SELECT u.id, u.email, u.license_key, u.license_type, u.license_status,
                       u.license_expiration, u.created_at,
                       MAX(a.created_at) as last_active
                FROM users u
                LEFT JOIN api_logs a ON u.license_key = a.license_key
                WHERE u.id = %s OR u.license_key = %s
                GROUP BY u.id, u.email, u.license_key, u.license_type, u.license_status,
                         u.license_expiration, u.created_at
            """, (account_id, account_id))
            user = cursor.fetchone()
            
            if not user:
                return jsonify({"error": "User not found"}), 404
            
            # Get API call count for this user
            cursor.execute("""
                SELECT COUNT(*) as api_calls
                FROM api_logs
                WHERE license_key = %s
            """, (user['license_key'],))
            api_call_result = cursor.fetchone()
            api_call_count = api_call_result['api_calls'] if api_call_result else 0
            
            # Get trade statistics for this user
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_trades,
                    COALESCE(SUM(pnl), 0) as total_pnl,
                    COALESCE(AVG(pnl), 0) as avg_pnl,
                    COUNT(*) FILTER (WHERE pnl > 0) as winning_trades
                FROM rl_experiences
                WHERE license_key = %s AND took_trade = TRUE
            """, (user['license_key'],))
            trade_stats_result = cursor.fetchone()
            
            # Format user data
            user_data = {
                "user": {
                    "account_id": str(user['id']),
                    "email": user['email'],
                    "license_key": user['license_key'],
                    "license_type": user['license_type'],
                    "license_status": user['license_status'],
                    "license_expiration": user['license_expiration'].isoformat() if user['license_expiration'] else None,
                    "created_at": user['created_at'].isoformat() if user['created_at'] else None,
                    "last_active": user['last_active'].isoformat() if user['last_active'] else None,
                    "notes": None
                },
                "recent_api_calls": api_call_count,
                "trade_stats": {
                    "total_trades": int(trade_stats_result['total_trades']) if trade_stats_result else 0,
                    "total_pnl": float(trade_stats_result['total_pnl']) if trade_stats_result else 0.0,
                    "avg_pnl": float(trade_stats_result['avg_pnl']) if trade_stats_result else 0.0,
                    "winning_trades": int(trade_stats_result['winning_trades']) if trade_stats_result else 0
                },
                "recent_activity": []
            }
            
            return jsonify(user_data), 200
    except Exception as e:
        logging.error(f"Get user error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/admin/recent-activity', methods=['GET'])
def admin_recent_activity():
    """Get recent API activity"""
    admin_key = request.args.get('license_key') or request.args.get('admin_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    limit = int(request.args.get('limit', 50))
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"activity": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    a.created_at as timestamp,
                    COALESCE(u.id::text, 'Unknown') as account_id,
                    a.endpoint,
                    'POST' as method,
                    a.status_code,
                    0 as response_time_ms,
                    '0.0.0.0' as ip_address
                FROM api_logs a
                LEFT JOIN users u ON a.license_key = u.license_key
                ORDER BY a.created_at DESC
                LIMIT %s
            """, (limit,))
            activity = cursor.fetchall()
            
            formatted_activity = []
            for act in activity:
                formatted_activity.append({
                    "timestamp": act['timestamp'].isoformat() if act['timestamp'] else None,
                    "account_id": act['account_id'],
                    "endpoint": act['endpoint'],
                    "method": act['method'],
                    "status_code": act['status_code'],
                    "response_time_ms": act['response_time_ms'],
                    "ip_address": act['ip_address']
                })
            
            return jsonify({"activity": formatted_activity}), 200
    except Exception as e:
        logging.error(f"Recent activity error: {e}")
        return jsonify({"activity": []}), 200
    finally:
        conn.close()

@app.route('/api/admin/online-users', methods=['GET'])
def admin_online_users():
    """Get currently online users"""
    admin_key = request.args.get('license_key') or request.args.get('admin_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"users": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT u.id, u.email, u.license_key, u.license_type, 
                       MAX(a.created_at) as last_active
                FROM users u
                INNER JOIN api_logs a ON u.license_key = a.license_key
                WHERE a.created_at > NOW() - INTERVAL '5 minutes'
                AND UPPER(u.license_status) = 'ACTIVE'
                GROUP BY u.id, u.email, u.license_key, u.license_type
                ORDER BY last_active DESC
            """)
            online = cursor.fetchall()
            
            formatted = []
            for user in online:
                formatted.append({
                    "account_id": str(user['id']),
                    "email": user['email'],
                    "license_key": user['license_key'],
                    "license_type": user['license_type'],
                    "last_active": user['last_active'].isoformat() if user['last_active'] else None
                })
            
            return jsonify({"users": formatted}), 200
    except Exception as e:
        logging.error(f"Online users error: {e}")
        return jsonify({"users": []}), 200
    finally:
        conn.close()

@app.route('/api/admin/suspend-user/<account_id>', methods=['POST', 'PUT'])
def admin_suspend_user(account_id):
    """Suspend a user (same as update-license-status)"""
    admin_key = request.args.get('license_key') or request.args.get('admin_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE users SET license_status = 'SUSPENDED'
                WHERE id = %s OR license_key = %s
                RETURNING license_key
            """, (account_id, account_id))
            result = cursor.fetchone()
            conn.commit()
            
            if result:
                return jsonify({"status": "success", "message": "User suspended"}), 200
            else:
                return jsonify({"error": "User not found"}), 404
    except Exception as e:
        logging.error(f"Suspend user error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/admin/activate-user/<account_id>', methods=['POST', 'PUT'])
def admin_activate_user(account_id):
    """Activate a user"""
    admin_key = request.args.get('license_key') or request.args.get('admin_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE users SET license_status = 'ACTIVE'
                WHERE id = %s OR license_key = %s
                RETURNING license_key
            """, (account_id, account_id))
            result = cursor.fetchone()
            conn.commit()
            
            if result:
                return jsonify({"status": "success", "message": "User activated"}), 200
            else:
                return jsonify({"error": "User not found"}), 404
    except Exception as e:
        logging.error(f"Activate user error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/admin/extend-license/<account_id>', methods=['POST', 'PUT'])
def admin_extend_license(account_id):
    """Extend a user's license"""
    admin_key = request.args.get('license_key') or request.args.get('admin_key')
    days = int(request.args.get('days', 30))
    
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                UPDATE users 
                SET license_expiration = COALESCE(license_expiration, NOW()) + INTERVAL '%s days'
                WHERE id = %s OR license_key = %s
                RETURNING license_key, license_expiration
            """, (days, account_id, account_id))
            result = cursor.fetchone()
            conn.commit()
            
            if result:
                return jsonify({
                    "status": "success", 
                    "message": f"License extended by {days} days",
                    "new_expiration": result[1].isoformat() if result[1] else None
                }), 200
            else:
                return jsonify({"error": "User not found"}), 404
    except Exception as e:
        logging.error(f"Extend license error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

@app.route('/api/admin/add-user', methods=['POST'])
def admin_add_user():
    """Create a new user (same as create-license but formatted for dashboard)"""
    admin_key = request.args.get('license_key') or request.args.get('admin_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    email = data.get('email')
    license_type = data.get('license_type', 'MONTHLY')
    days_valid = data.get('days_valid', 30)
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"error": "Database connection failed"}), 500
    
    try:
        license_key = generate_license_key()
        expiration = datetime.now() + timedelta(days=days_valid)
        
        with conn.cursor() as cursor:
            cursor.execute("""
                INSERT INTO users (email, license_key, license_type, license_status, license_expiration)
                VALUES (%s, %s, %s, 'ACTIVE', %s)
                RETURNING id
            """, (email, license_key, license_type, expiration))
            user_id = cursor.fetchone()[0]
            conn.commit()
            
            return jsonify({
                "status": "success",
                "license_key": license_key,
                "account_id": str(user_id),
                "email": email,
                "expiration": expiration.isoformat()
            }), 201
    except Exception as e:
        logging.error(f"Add user error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        conn.close()

# ============================================================================
# END ADMIN DASHBOARD ENDPOINTS
# ============================================================================

@app.route('/api/admin/expire-licenses', methods=['POST'])
def expire_licenses():
    """Manually trigger license expiration check (can be called by cron/scheduler)"""
    try:
        api_key = request.headers.get('X-Admin-Key')
        if api_key != ADMIN_API_KEY:
            return jsonify({"status": "error", "message": "Unauthorized"}), 401
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"status": "error", "message": "Database error"}), 500
        
        try:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE users 
                    SET license_status = 'expired'
                    WHERE license_expiration < NOW() 
                    AND license_status = 'active'
                    RETURNING license_key, email
                """)
                expired = cursor.fetchall()
                conn.commit()
                
                return jsonify({
                    "status": "success",
                    "expired_count": len(expired),
                    "expired_licenses": [row[0] for row in expired]
                }), 200
        finally:
            conn.close()
            
    except Exception as e:
        logging.error(f"Error expiring licenses: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

# ============================================================================
# RL EXPERIENCE ENDPOINTS - Centralized Learning System
# ============================================================================
# Note: RL experiences are now stored in PostgreSQL (rl_experiences table)
# Bots write experiences directly to the database via the bot SDK
# Admin endpoints query the database for monitoring and analytics

@app.route('/api/admin/rl-stats', methods=['GET'])
def admin_rl_stats():
    """Admin endpoint to view RL statistics"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        # Return default values if database is not available
        return jsonify({
            "total_experiences": 0,
            "win_rate": 0.0,
            "avg_reward": 0.0,
            "total_reward": 0.0,
            "last_updated": datetime.now().isoformat()
        }), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get RL statistics from database
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE pnl > 0) as winning,
                    SUM(pnl) as total_reward,
                    AVG(pnl) as avg_reward,
                    MAX(created_at) as last_updated
                FROM rl_experiences
                WHERE took_trade = TRUE
            """)
            stats = cursor.fetchone()
            
            total = int(stats['total']) if stats['total'] else 0
            winning = int(stats['winning']) if stats['winning'] else 0
            win_rate = (winning / total * 100) if total > 0 else 0.0
            total_reward = float(stats['total_reward']) if stats['total_reward'] else 0.0
            avg_reward = float(stats['avg_reward']) if stats['avg_reward'] else 0.0
            last_updated = stats['last_updated'].isoformat() if stats['last_updated'] else datetime.now().isoformat()
            
            return jsonify({
                "total_experiences": total,
                "win_rate": win_rate,
                "avg_reward": avg_reward,
                "total_reward": total_reward,
                "last_updated": last_updated
            }), 200
    except Exception as e:
        logging.error(f"RL stats error: {e}")
        # Return default values on error
        return jsonify({
            "total_experiences": 0,
            "win_rate": 0.0,
            "avg_reward": 0.0,
            "total_reward": 0.0,
            "last_updated": datetime.now().isoformat()
        }), 200
    finally:
        conn.close()

@app.route('/api/admin/rl-experiences', methods=['GET'])
def admin_rl_experiences():
    """Admin endpoint to view recent RL experiences with full details"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    limit = int(request.args.get('limit', 100))
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"experiences": [], "total_experiences": 0, "limit": limit}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get total count
            cursor.execute("SELECT COUNT(*) as total FROM rl_experiences")
            total = cursor.fetchone()['total']
            
            # Get recent experiences
            cursor.execute("""
                SELECT 
                    created_at as timestamp,
                    symbol,
                    rsi,
                    vwap_distance,
                    atr,
                    volume_ratio,
                    hour,
                    day_of_week,
                    recent_pnl,
                    streak,
                    side,
                    regime,
                    took_trade,
                    pnl as reward,
                    duration,
                    0.0 as price
                FROM rl_experiences
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            experiences = cursor.fetchall()
            
            formatted_experiences = []
            for exp in experiences:
                formatted_experiences.append({
                    "timestamp": exp['timestamp'].isoformat() if exp['timestamp'] else None,
                    "state": {
                        "symbol": exp['symbol'],
                        "rsi": float(exp['rsi']) if exp['rsi'] else 0,
                        "vwap_distance": float(exp['vwap_distance']) if exp['vwap_distance'] else 0,
                        "atr": float(exp['atr']) if exp['atr'] else 0,
                        "volume_ratio": float(exp['volume_ratio']) if exp['volume_ratio'] else 0,
                        "hour": int(exp['hour']) if exp['hour'] else 0,
                        "day_of_week": int(exp['day_of_week']) if exp['day_of_week'] else 0,
                        "recent_pnl": float(exp['recent_pnl']) if exp['recent_pnl'] else 0,
                        "streak": int(exp['streak']) if exp['streak'] else 0,
                        "side": exp['side'],
                        "regime": exp['regime'],
                        "price": float(exp['price']) if exp['price'] else 0
                    },
                    "action": {
                        "took_trade": bool(exp['took_trade'])
                    },
                    "reward": float(exp['reward']) if exp['reward'] else 0,
                    "duration": float(exp['duration']) if exp['duration'] else 0
                })
            
            return jsonify({
                "total_experiences": total,
                "experiences": formatted_experiences,
                "limit": limit
            }), 200
    except Exception as e:
        logging.error(f"RL experiences error: {e}")
        return jsonify({"experiences": [], "total_experiences": 0, "limit": limit}), 200
    finally:
        conn.close()

# ============================================================================
# RL DECISION ENDPOINTS - Cloud makes trading decisions for user bots
# ============================================================================

@app.route('/api/rl/analyze-signal', methods=['POST'])
def analyze_signal():
    """
    User bot sends market state, cloud RL brain decides whether to take trade.
    
    Request body:
    {
        "license_key": "user's license key",
        "state": {
            "rsi": 45.2,
            "vwap_distance": 0.02,
            "atr": 2.5,
            "volume_ratio": 1.3,
            "hour": 14,
            "day_of_week": 2,
            "recent_pnl": -50.0,
            "streak": -1,
            "side": "long",
            "price": 6767.75
        }
    }
    
    Response:
    {
        "take_trade": true/false,
        "confidence": 0.68,
        "reason": "✅ TAKE (68% confidence) - 8W/2L similar | Winners: 75% WR, $150 avg"
    }
    """
    try:
        data = request.get_json()
        license_key = data.get('license_key')
        state = data.get('state', {})
        
        # Validate license key
        if not license_key:
            return jsonify({"error": "Missing license_key"}), 401
        
        # Check license in database
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database unavailable"}), 503
        
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        try:
            cursor.execute("""
                SELECT license_key, status, expires_at 
                FROM users 
                WHERE license_key = %s
            """, (license_key,))
            
            license_data = cursor.fetchone()
            
            if not license_data:
                return jsonify({"error": "Invalid license key"}), 401
            
            if license_data['status'] != 'active':
                return jsonify({"error": f"License is {license_data['status']}"}), 401
            
            if license_data['expires_at'] and datetime.now() > license_data['expires_at']:
                return jsonify({"error": "License expired"}), 401
            
            # Load RL brain and make decision for this symbol
            symbol = state.get('symbol', 'ES')
            experiences = load_experiences(symbol)
            
            if experiences is None:
                # Database connection failed - REJECT for safety
                return jsonify({
                    "take_trade": False,
                    "confidence": 0.0,
                    "reason": "❌ Database unavailable - rejecting for safety"
                }), 200
            
            if not experiences:
                # No historical data for this symbol - REJECT for safety
                return jsonify({
                    "take_trade": False,
                    "confidence": 0.0,
                    "reason": "❌ No historical data for symbol - rejecting for safety"
                }), 200
            
            # Create decision engine and analyze
            engine = CloudRLDecisionEngine(experiences)
            take_trade, confidence, reason = engine.should_take_signal(state)
            
            # Log API call
            cursor.execute("""
                INSERT INTO api_logs (license_key, endpoint, created_at)
                VALUES (%s, %s, NOW())
            """, (license_key, 'rl/analyze-signal'))
            conn.commit()
            
            return jsonify({
                "take_trade": take_trade,
                "confidence": confidence,
                "reason": reason
            }), 200
                
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        logging.error(f"Analyze signal error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/rl/submit-outcome', methods=['POST'])
def submit_outcome():
    """
    User bot reports trade outcome after execution (win/loss).
    Cloud RL brain records this to PostgreSQL for scalability.
    
    SCALES TO 1000+ USERS: Direct database insert, no locking needed.
    PostgreSQL handles concurrent writes natively.
    
    Request body:
    {
        "license_key": "user's license key",
        "state": {...},  // Same state sent to analyze-signal
        "took_trade": true,
        "pnl": 125.50,
        "duration": 1800  // seconds
    }
    
    Response:
    {
        "success": true,
        "total_experiences": 7560,
        "win_rate": 0.56
    }
    """
    try:
        data = request.get_json()
        license_key = data.get('license_key')
        state = data.get('state', {})
        took_trade = data.get('took_trade', False)
        pnl = data.get('pnl', 0.0)
        duration = data.get('duration', 0.0)
        
        # Validate license key
        if not license_key:
            return jsonify({"error": "Missing license_key"}), 401
        
        conn = get_db_connection()
        if not conn:
            return jsonify({"error": "Database unavailable"}), 503
        
        try:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("""
                SELECT license_key, status, expires_at 
                FROM users 
                WHERE license_key = %s
            """, (license_key,))
            
            license_data = cursor.fetchone()
            
            if not license_data:
                return jsonify({"error": "Invalid license key"}), 401
            
            if license_data['status'] != 'active':
                return jsonify({"error": f"License is {license_data['status']}"}), 401
            
            # Insert outcome directly into PostgreSQL (instant, no locking)
            cursor.execute("""
                INSERT INTO rl_experiences (
                    license_key,
                    symbol,
                    rsi,
                    vwap_distance,
                    atr,
                    volume_ratio,
                    hour,
                    day_of_week,
                    recent_pnl,
                    streak,
                    side,
                    regime,
                    took_trade,
                    pnl,
                    duration,
                    created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """, (
                license_key,
                state.get('symbol', 'ES'),  # Default to ES if not provided
                state.get('rsi', 50.0),
                state.get('vwap_distance', 0.0),
                state.get('atr', 0.0),
                state.get('volume_ratio', 1.0),
                state.get('hour', 0),
                state.get('day_of_week', 0),
                state.get('recent_pnl', 0.0),
                state.get('streak', 0),
                state.get('side', 'long'),
                state.get('regime', 'NORMAL'),
                took_trade,
                pnl,
                duration
            ))
            
            # Get total experiences and win rate for this symbol
            symbol = state.get('symbol', 'ES')
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    AVG(CASE WHEN took_trade AND pnl > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                    AVG(CASE WHEN took_trade THEN pnl ELSE 0 END) as avg_reward
                FROM rl_experiences
                WHERE took_trade = TRUE AND symbol = %s
            """, (symbol,))
            stats = cursor.fetchone()
            
            # Log API call
            cursor.execute("""
                INSERT INTO api_logs (license_key, endpoint, created_at)
                VALUES (%s, %s, NOW())
            """, (license_key, 'rl/submit-outcome'))
            
            conn.commit()
            
            # Clear cache so next load gets fresh data from database
            global _experiences_cache, _experiences_cache_time
            _experiences_cache = None
            _experiences_cache_time = None
            
            return jsonify({
                "success": True,
                "total_experiences": int(stats['total'] or 0),
                "win_rate": float(stats['win_rate'] or 0.0),
                "avg_reward": float(stats['avg_reward'] or 0.0)
            }), 200
                
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        logging.error(f"Submit outcome error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# END RL DECISION ENDPOINTS
# ============================================================================

# ============================================================================
# RL ADMIN/STATS ENDPOINTS (existing)
# ============================================================================

@app.route('/api/stripe/success', methods=['GET'])
def stripe_success():
    """Stripe checkout success page"""
    session_id = request.args.get('session_id')
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Payment Successful - QuoTrading</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                   display: flex; justify-content: center; align-items: center; 
                   height: 100vh; margin: 0; }}
            .container {{ background: white; padding: 60px; border-radius: 20px; 
                         box-shadow: 0 20px 60px rgba(0,0,0,0.3); text-align: center; 
                         max-width: 500px; }}
            h1 {{ color: #10b981; font-size: 48px; margin: 0 0 20px 0; }}
            p {{ color: #6b7280; font-size: 18px; line-height: 1.6; }}
            .check {{ font-size: 80px; color: #10b981; margin-bottom: 20px; }}
            .session {{ background: #f3f4f6; padding: 15px; border-radius: 10px; 
                       font-family: monospace; font-size: 12px; margin-top: 20px; 
                       color: #6b7280; word-break: break-all; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="check">✓</div>
            <h1>Payment Successful!</h1>
            <p>Thank you for subscribing to QuoTrading Bot!</p>
            <p>Your license key will be emailed to you within a few minutes.</p>
            <p style="margin-top: 30px; font-size: 16px; color: #9ca3af;">
                Check your inbox for setup instructions and your license key.
            </p>
            <div class="session">Session: {session_id}</div>
        </div>
    </body>
    </html>
    """

@app.route('/api/stripe/cancel', methods=['GET'])
def stripe_cancel():
    """Stripe checkout cancelled page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Payment Cancelled - QuoTrading</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                   background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                   display: flex; justify-content: center; align-items: center; 
                   height: 100vh; margin: 0; }
            .container { background: white; padding: 60px; border-radius: 20px; 
                         box-shadow: 0 20px 60px rgba(0,0,0,0.3); text-align: center; 
                         max-width: 500px; }
            h1 { color: #ef4444; font-size: 48px; margin: 0 0 20px 0; }
            p { color: #6b7280; font-size: 18px; line-height: 1.6; }
            .icon { font-size: 80px; color: #ef4444; margin-bottom: 20px; }
            a { display: inline-block; margin-top: 30px; padding: 15px 40px; 
                background: #667eea; color: white; text-decoration: none; 
                border-radius: 10px; font-weight: 600; transition: all 0.3s; }
            a:hover { background: #5568d3; transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="icon">✕</div>
            <h1>Payment Cancelled</h1>
            <p>Your payment was cancelled. No charges were made.</p>
            <p style="margin-top: 20px; color: #9ca3af;">
                If you have any questions, please contact support.
            </p>
            <a href="https://discord.gg/quotrading">Return to Discord</a>
        </div>
    </body>
    </html>
    """

@app.route('/', methods=['GET'])
def root():
    """Root endpoint"""
    return jsonify({
        "status": "success",
        "message": "QuoTrading API",
        "endpoints": ["/api/hello", "/api/main", "/api/stripe/webhook", "/api/admin/create-license", "/api/admin/expire-licenses"]
    }), 200

# ========== PHASE 2: CHART DATA ENDPOINTS ==========

@app.route('/api/admin/charts/user-growth', methods=['GET'])
def admin_chart_user_growth():
    """Get user growth by week for last 12 weeks"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"weeks": [], "counts": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    DATE_TRUNC('week', created_at) as week,
                    COUNT(*) as count
                FROM users
                WHERE created_at >= NOW() - INTERVAL '12 weeks'
                GROUP BY week
                ORDER BY week
            """)
            results = cursor.fetchall()
            
            weeks = [f"Week {i+1}" for i in range(len(results))]
            counts = [int(r['count']) for r in results]
            
            return jsonify({"weeks": weeks, "counts": counts}), 200
    except Exception as e:
        logging.error(f"User growth chart error: {e}")
        return jsonify({"weeks": [], "counts": []}), 200
    finally:
        conn.close()

@app.route('/api/admin/charts/api-usage', methods=['GET'])
def admin_chart_api_usage():
    """Get API calls per hour for last 24 hours"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"hours": [], "counts": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    EXTRACT(HOUR FROM timestamp) as hour,
                    COUNT(*) as count
                FROM api_logs
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
                GROUP BY hour
                ORDER BY hour
            """)
            results = cursor.fetchall()
            
            # Create 24-hour array with 0 for missing hours
            hour_counts = {int(r['hour']): int(r['count']) for r in results}
            hours = [f"{h:02d}:00" for h in range(24)]
            counts = [hour_counts.get(h, 0) for h in range(24)]
            
            return jsonify({"hours": hours, "counts": counts}), 200
    except Exception as e:
        logging.error(f"API usage chart error: {e}")
        return jsonify({"hours": [], "counts": []}), 200
    finally:
        conn.close()

@app.route('/api/admin/charts/mrr', methods=['GET'])
def admin_chart_mrr():
    """Get Monthly Recurring Revenue trend"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"months": [], "revenue": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    TO_CHAR(DATE_TRUNC('month', created_at), 'Mon') as month,
                    COUNT(*) FILTER (WHERE license_type = 'MONTHLY') * 200.00 +
                    COUNT(*) FILTER (WHERE license_type = 'ANNUAL') * 2000.00 as revenue
                FROM users
                WHERE created_at >= NOW() - INTERVAL '6 months'
                AND UPPER(license_status) = 'ACTIVE'
                GROUP BY DATE_TRUNC('month', created_at)
                ORDER BY DATE_TRUNC('month', created_at)
            """)
            results = cursor.fetchall()
            
            months = [r['month'] for r in results]
            revenue = [float(r['revenue']) if r['revenue'] else 0 for r in results]
            
            return jsonify({"months": months, "revenue": revenue}), 200
    except Exception as e:
        logging.error(f"MRR chart error: {e}")
        return jsonify({"months": [], "revenue": []}), 200
    finally:
        conn.close()

@app.route('/api/admin/charts/collective-pnl', methods=['GET'])
def admin_chart_collective_pnl():
    """Get collective P&L for all users daily (last 30 days)"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"dates": [], "pnl": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    DATE(created_at) as date,
                    SUM(pnl) as daily_pnl
                FROM rl_experiences
                WHERE created_at >= NOW() - INTERVAL '30 days'
                AND took_trade = TRUE
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at)
            """)
            results = cursor.fetchall()
            
            # Calculate cumulative P&L
            cumulative = 0
            dates = []
            pnl_values = []
            
            for r in results:
                cumulative += float(r['daily_pnl']) if r['daily_pnl'] else 0
                dates.append(r['date'].strftime('%b %d'))
                pnl_values.append(round(cumulative, 2))
            
            return jsonify({"dates": dates, "pnl": pnl_values}), 200
    except Exception as e:
        logging.error(f"Collective P&L chart error: {e}")
        return jsonify({"dates": [], "pnl": []}), 200
    finally:
        conn.close()

@app.route('/api/admin/charts/win-rate-trend', methods=['GET'])
def admin_chart_win_rate_trend():
    """Get win rate trend by week"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"weeks": [], "win_rates": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    DATE_TRUNC('week', created_at) as week,
                    COUNT(*) FILTER (WHERE pnl > 0) * 100.0 / NULLIF(COUNT(*), 0) as win_rate
                FROM rl_experiences
                WHERE created_at >= NOW() - INTERVAL '12 weeks'
                AND took_trade = TRUE
                GROUP BY week
                ORDER BY week
            """)
            results = cursor.fetchall()
            
            weeks = [f"Week {i+1}" for i in range(len(results))]
            win_rates = [round(float(r['win_rate']), 2) if r['win_rate'] else 0 for r in results]
            
            return jsonify({"weeks": weeks, "win_rates": win_rates}), 200
    except Exception as e:
        logging.error(f"Win rate trend chart error: {e}")
        return jsonify({"weeks": [], "win_rates": []}), 200
    finally:
        conn.close()

@app.route('/api/admin/charts/top-performers', methods=['GET'])
def admin_chart_top_performers():
    """Get top 10 users by P&L"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"users": [], "pnl": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    license_key,
                    SUM(pnl) as total_pnl
                FROM rl_experiences
                WHERE took_trade = TRUE
                GROUP BY license_key
                ORDER BY total_pnl DESC
                LIMIT 10
            """)
            results = cursor.fetchall()
            
            users = [r['license_key'][:12] + '...' for r in results]  # Truncate long keys
            pnl = [round(float(r['total_pnl']), 2) if r['total_pnl'] else 0 for r in results]
            
            return jsonify({"users": users, "pnl": pnl}), 200
    except Exception as e:
        logging.error(f"Top performers chart error: {e}")
        return jsonify({"users": [], "pnl": []}), 200
    finally:
        conn.close()

@app.route('/api/admin/charts/experience-growth', methods=['GET'])
def admin_chart_experience_growth():
    """Get experience accumulation over time (last 30 days)"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"dates": [], "counts": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as daily_count
                FROM rl_experiences
                WHERE created_at >= NOW() - INTERVAL '30 days'
                GROUP BY DATE(created_at)
                ORDER BY DATE(created_at)
            """)
            results = cursor.fetchall()
            
            # Calculate cumulative count
            cumulative = 0
            dates = []
            counts = []
            
            for r in results:
                cumulative += int(r['daily_count'])
                dates.append(r['date'].strftime('%b %d'))
                counts.append(cumulative)
            
            return jsonify({"dates": dates, "counts": counts}), 200
    except Exception as e:
        logging.error(f"Experience growth chart error: {e}")
        return jsonify({"dates": [], "counts": []}), 200
    finally:
        conn.close()

@app.route('/api/admin/charts/confidence-dist', methods=['GET'])
def admin_chart_confidence_dist():
    """Get confidence level distribution (histogram)"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"ranges": [], "counts": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Assuming confidence can be derived from took_trade probability
            # For now, create mock distribution based on trade patterns
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN ABS(recent_pnl) < 10 THEN '0-20%'
                        WHEN ABS(recent_pnl) < 20 THEN '20-40%'
                        WHEN ABS(recent_pnl) < 30 THEN '40-60%'
                        WHEN ABS(recent_pnl) < 40 THEN '60-80%'
                        ELSE '80-100%'
                    END as confidence_range,
                    COUNT(*) as count
                FROM rl_experiences
                GROUP BY confidence_range
                ORDER BY confidence_range
            """)
            results = cursor.fetchall()
            
            ranges = [r['confidence_range'] for r in results]
            counts = [int(r['count']) for r in results]
            
            return jsonify({"ranges": ranges, "counts": counts}), 200
    except Exception as e:
        logging.error(f"Confidence distribution chart error: {e}")
        return jsonify({"ranges": [], "counts": []}), 200
    finally:
        conn.close()

@app.route('/api/admin/charts/confidence-winrate', methods=['GET'])
def admin_chart_confidence_winrate():
    """Get win rate by confidence level (scatter plot data)"""
    admin_key = request.args.get('admin_key') or request.args.get('license_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    if not conn:
        return jsonify({"confidence": [], "win_rate": [], "sample_size": []}), 200
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT 
                    CASE 
                        WHEN ABS(recent_pnl) < 10 THEN 10
                        WHEN ABS(recent_pnl) < 20 THEN 30
                        WHEN ABS(recent_pnl) < 30 THEN 50
                        WHEN ABS(recent_pnl) < 40 THEN 70
                        ELSE 90
                    END as confidence_level,
                    COUNT(*) FILTER (WHERE pnl > 0) * 100.0 / NULLIF(COUNT(*), 0) as win_rate,
                    COUNT(*) as sample_size
                FROM rl_experiences
                WHERE took_trade = TRUE
                GROUP BY confidence_level
                ORDER BY confidence_level
            """)
            results = cursor.fetchall()
            
            confidence = [int(r['confidence_level']) for r in results]
            win_rate = [round(float(r['win_rate']), 2) if r['win_rate'] else 0 for r in results]
            sample_size = [int(r['sample_size']) for r in results]
            
            return jsonify({
                "confidence": confidence,
                "win_rate": win_rate,
                "sample_size": sample_size
            }), 200
    except Exception as e:
        logging.error(f"Confidence vs win rate chart error: {e}")
        return jsonify({"confidence": [], "win_rate": [], "sample_size": []}), 200
    finally:
        conn.close()

# ==================== REPORTS ENDPOINTS ====================

@app.route('/api/admin/reports/user-activity', methods=['GET'])
def admin_report_user_activity():
    """Generate user activity report with date range filters"""
    auth_header = request.headers.get('X-API-Key')
    if auth_header != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    # Get query parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    license_type = request.args.get('license_type', 'all')
    status = request.args.get('status', 'all')
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT 
                l.account_id,
                l.email,
                l.created_at,
                l.last_active,
                l.license_type,
                l.license_status,
                COUNT(DISTINCT a.id) as api_calls,
                COUNT(DISTINCT r.id) FILTER (WHERE r.took_trade = TRUE) as trades,
                COALESCE(SUM(r.pnl), 0) as total_pnl
            FROM users l
            LEFT JOIN api_logs a ON l.license_key = a.license_key
            LEFT JOIN rl_experiences r ON l.license_key = r.license_key AND r.took_trade = TRUE
            WHERE 1=1
        """
        params = []
        
        if start_date:
            query += " AND l.created_at >= %s"
            params.append(start_date)
        if end_date:
            query += " AND l.created_at <= %s"
            params.append(end_date)
        if license_type != 'all':
            query += " AND UPPER(l.license_type) = UPPER(%s)"
            params.append(license_type)
        if status != 'all':
            query += " AND UPPER(l.license_status) = UPPER(%s)"
            params.append(status)
        
        query += " GROUP BY l.account_id, l.email, l.created_at, l.last_active, l.license_type, l.license_status"
        query += " ORDER BY l.created_at DESC LIMIT 500"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        # Format results
        formatted_results = []
        for r in results:
            formatted_results.append({
                "account_id": r['account_id'][:8] + "..." if r['account_id'] else "N/A",
                "email": r['email'],
                "signup_date": r['created_at'].strftime('%Y-%m-%d') if r['created_at'] else "N/A",
                "last_active": r['last_active'].strftime('%Y-%m-%d %H:%M') if r['last_active'] else "Never",
                "license_type": r['license_type'],
                "status": r['license_status'],
                "api_calls": int(r['api_calls']),
                "trades": int(r['trades']),
                "total_pnl": round(float(r['total_pnl']), 2)
            })
        
        return jsonify({"data": formatted_results, "count": len(formatted_results)}), 200
    except Exception as e:
        logging.error(f"User activity report error: {e}")
        return jsonify({"error": str(e), "data": [], "count": 0}), 200
    finally:
        conn.close()

@app.route('/api/admin/reports/revenue', methods=['GET'])
def admin_report_revenue():
    """Generate revenue analysis report"""
    auth_header = request.headers.get('X-API-Key')
    if auth_header != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    month = request.args.get('month', str(datetime.now().month))
    year = request.args.get('year', str(datetime.now().year))
    license_type_filter = request.args.get('license_type', 'all')
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Define pricing
        pricing = {
            'MONTHLY': 200.00,
            'ANNUAL': 2000.00,
            'TRIAL': 0.00
        }
        
        # Get new subscriptions
        query_new = """
            SELECT COUNT(*) as count, UPPER(license_type) as type
            FROM users
            WHERE EXTRACT(MONTH FROM created_at) = %s
              AND EXTRACT(YEAR FROM created_at) = %s
        """
        params = [int(month), int(year)]
        
        if license_type_filter != 'all':
            query_new += " AND UPPER(license_type) = UPPER(%s)"
            params.append(license_type_filter)
        
        query_new += " GROUP BY UPPER(license_type)"
        cursor.execute(query_new, params)
        new_subs = cursor.fetchall()
        
        # Calculate metrics
        new_count = sum(r['count'] for r in new_subs)
        new_revenue = sum(r['count'] * pricing.get(r['type'], 0) for r in new_subs)
        
        # Get active users
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM users
            WHERE UPPER(license_status) = 'ACTIVE'
        """)
        active_users = cursor.fetchone()['count']
        
        # Get expired this month
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM users
            WHERE license_expiration >= %s
              AND license_expiration < %s
              AND EXTRACT(MONTH FROM license_expiration) = %s
              AND EXTRACT(YEAR FROM license_expiration) = %s
        """, [
            f"{year}-{month}-01",
            f"{year}-{int(month)+1 if int(month) < 12 else 1}-01",
            int(month),
            int(year)
        ])
        expired = cursor.fetchone()['count']
        
        # Calculate MRR (all active monthly licenses)
        cursor.execute("""
            SELECT COUNT(*) as count, UPPER(license_type) as type
            FROM users
            WHERE UPPER(license_status) = 'ACTIVE'
            GROUP BY UPPER(license_type)
        """)
        active_breakdown = cursor.fetchall()
        
        mrr = sum(r['count'] * (pricing.get(r['type'], 0) if r['type'] == 'MONTHLY' else pricing.get(r['type'], 0) / 12) for r in active_breakdown)
        arpu = mrr / active_users if active_users > 0 else 0
        churn_rate = (expired / active_users * 100) if active_users > 0 else 0
        
        return jsonify({
            "new_subscriptions": new_count,
            "new_revenue": round(new_revenue, 2),
            "renewals": 0,  # Would need renewal tracking
            "renewal_revenue": 0.00,
            "cancellations": expired,
            "lost_revenue": round(expired * 200.00, 2),  # Estimate
            "net_mrr": round(mrr, 2),
            "churn_rate": round(churn_rate, 2),
            "arpu": round(arpu, 2),
            "active_users": active_users
        }), 200
    except Exception as e:
        logging.error(f"Revenue report error: {e}")
        return jsonify({"error": str(e)}), 200
    finally:
        conn.close()

@app.route('/api/admin/reports/performance', methods=['GET'])
def admin_report_performance():
    """Generate trading performance report"""
    auth_header = request.headers.get('X-API-Key')
    if auth_header != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    symbol = request.args.get('symbol', 'all')
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        query = """
            SELECT 
                COUNT(*) as total_trades,
                AVG(CASE WHEN pnl > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate,
                SUM(pnl) as total_pnl,
                AVG(confidence) as avg_confidence,
                AVG(EXTRACT(EPOCH FROM (exit_time - entry_time)) / 60) as avg_duration_minutes
            FROM rl_experiences
            WHERE took_trade = TRUE
        """
        params = []
        
        if start_date:
            query += " AND timestamp >= %s"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= %s"
            params.append(end_date)
        if symbol != 'all':
            query += " AND symbol = %s"
            params.append(symbol)
        
        cursor.execute(query, params)
        result = cursor.fetchone()
        
        # Get best and worst days
        day_query = """
            SELECT 
                DATE(timestamp) as trade_date,
                SUM(pnl) as daily_pnl
            FROM rl_experiences
            WHERE took_trade = TRUE
        """
        if start_date:
            day_query += " AND timestamp >= %s"
        if end_date:
            day_query += " AND timestamp <= %s"
        if symbol != 'all':
            day_query += " AND symbol = %s"
        
        day_query += " GROUP BY DATE(timestamp) ORDER BY daily_pnl DESC LIMIT 1"
        cursor.execute(day_query, params)
        best_day = cursor.fetchone()
        
        day_query = day_query.replace("DESC", "ASC")
        cursor.execute(day_query, params)
        worst_day = cursor.fetchone()
        
        return jsonify({
            "total_trades": int(result['total_trades']) if result['total_trades'] else 0,
            "win_rate": round(float(result['win_rate']), 2) if result['win_rate'] else 0,
            "total_pnl": round(float(result['total_pnl']), 2) if result['total_pnl'] else 0,
            "avg_confidence": round(float(result['avg_confidence']), 2) if result['avg_confidence'] else 0,
            "avg_duration_minutes": round(float(result['avg_duration_minutes']), 2) if result['avg_duration_minutes'] else 0,
            "best_day": {
                "date": best_day['trade_date'].strftime('%Y-%m-%d') if best_day else "N/A",
                "pnl": round(float(best_day['daily_pnl']), 2) if best_day else 0
            },
            "worst_day": {
                "date": worst_day['trade_date'].strftime('%Y-%m-%d') if worst_day else "N/A",
                "pnl": round(float(worst_day['daily_pnl']), 2) if worst_day else 0
            }
        }), 200
    except Exception as e:
        logging.error(f"Performance report error: {e}")
        return jsonify({"error": str(e)}), 200
    finally:
        conn.close()

@app.route('/api/admin/reports/retention', methods=['GET'])
def admin_report_retention():
    """Generate retention and churn report"""
    auth_header = request.headers.get('X-API-Key')
    if auth_header != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Current active users
        cursor.execute("SELECT COUNT(*) as count FROM users WHERE UPPER(license_status) = 'ACTIVE'")
        active_users = cursor.fetchone()['count']
        
        # Expired this month
        cursor.execute("""
            SELECT COUNT(*) as count FROM users
            WHERE license_expiration >= DATE_TRUNC('month', NOW())
              AND license_expiration < DATE_TRUNC('month', NOW()) + INTERVAL '1 month'
        """)
        expired_this_month = cursor.fetchone()['count']
        
        # Average subscription length
        cursor.execute("""
            SELECT AVG(EXTRACT(DAY FROM license_expiration - created_at)) as avg_days
            FROM users
            WHERE license_expiration IS NOT NULL
        """)
        avg_length = cursor.fetchone()['avg_days']
        
        # Cohort analysis - users by signup month
        cursor.execute("""
            SELECT 
                TO_CHAR(created_at, 'YYYY-MM') as cohort_month,
                COUNT(*) as users,
                COUNT(*) FILTER (WHERE UPPER(license_status) = 'ACTIVE') as still_active
            FROM users
            WHERE created_at >= NOW() - INTERVAL '12 months'
            GROUP BY TO_CHAR(created_at, 'YYYY-MM')
            ORDER BY cohort_month DESC
            LIMIT 12
        """)
        cohorts = cursor.fetchall()
        
        # Calculate metrics
        renewals = active_users - expired_this_month if active_users > 0 else 0
        retention_rate = (renewals / active_users * 100) if active_users > 0 else 0
        churn_rate = (expired_this_month / active_users * 100) if active_users > 0 else 0
        
        # Lifetime value (average)
        ltv = (avg_length / 30 * 200.00) if avg_length else 0
        
        cohort_data = []
        for c in cohorts:
            retention = (c['still_active'] / c['users'] * 100) if c['users'] > 0 else 0
            cohort_data.append({
                "month": c['cohort_month'],
                "users": c['users'],
                "still_active": c['still_active'],
                "retention": round(retention, 2)
            })
        
        return jsonify({
            "active_users": active_users,
            "expired_this_month": expired_this_month,
            "renewals": renewals,
            "retention_rate": round(retention_rate, 2),
            "churn_rate": round(churn_rate, 2),
            "avg_subscription_days": round(float(avg_length), 2) if avg_length else 0,
            "lifetime_value": round(ltv, 2),
            "cohorts": cohort_data
        }), 200
    except Exception as e:
        logging.error(f"Retention report error: {e}")
        return jsonify({"error": str(e)}), 200
    finally:
        conn.close()

def init_database_if_needed():
    """Initialize database table and indexes if they don't exist"""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode='require'
        )
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rl_experiences (
                id SERIAL PRIMARY KEY,
                license_key VARCHAR(50) NOT NULL,
                symbol VARCHAR(20) NOT NULL,
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
                took_trade BOOLEAN NOT NULL,
                pnl DECIMAL(10,2) NOT NULL,
                duration DECIMAL(10,2) NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Create indexes if not exist
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_rl_experiences_license ON rl_experiences(license_key)",
            "CREATE INDEX IF NOT EXISTS idx_rl_experiences_symbol ON rl_experiences(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_rl_experiences_created ON rl_experiences(created_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_rl_experiences_took_trade ON rl_experiences(took_trade)",
            "CREATE INDEX IF NOT EXISTS idx_rl_experiences_side ON rl_experiences(side)",
            "CREATE INDEX IF NOT EXISTS idx_rl_experiences_regime ON rl_experiences(regime)",
            "CREATE INDEX IF NOT EXISTS idx_rl_experiences_similarity ON rl_experiences(symbol, rsi, vwap_distance, atr, volume_ratio, side, regime)"
        ]
        
        for idx in indexes:
            cursor.execute(idx)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        app.logger.info("✅ PostgreSQL database initialized successfully")
        
    except Exception as e:
        app.logger.warning(f"Database initialization check: {e}")

@app.route('/api/admin/system-health', methods=['GET'])
def admin_system_health():
    """Get system health status for monitoring"""
    admin_key = request.args.get('license_key') or request.args.get('admin_key')
    if admin_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    health_status = {
        "timestamp": datetime.now().isoformat(),
        "database": {"status": "unknown", "response_time_ms": 0, "error": None},
        "rl_engine": {"status": "unknown", "total_experiences": 0, "response_time_ms": 0, "error": None}
    }
    
    # Test PostgreSQL connection
    db_start = datetime.now()
    try:
        conn = get_db_connection()
        if conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM rl_experiences WHERE took_trade = TRUE")
                result = cursor.fetchone()
                total_experiences = result['count'] if result else 0
                
            conn.close()
            db_time = (datetime.now() - db_start).total_seconds() * 1000
            health_status["database"] = {
                "status": "healthy",
                "response_time_ms": round(db_time, 2),
                "error": None
            }
            health_status["rl_engine"] = {
                "status": "healthy",
                "total_experiences": total_experiences,
                "response_time_ms": round(db_time, 2),
                "last_query": datetime.now().isoformat(),
                "error": None
            }
        else:
            health_status["database"] = {
                "status": "unhealthy",
                "response_time_ms": 0,
                "error": "Database connection failed"
            }
            health_status["rl_engine"] = {
                "status": "unhealthy",
                "total_experiences": 0,
                "response_time_ms": 0,
                "error": "Cannot query RL data - database unavailable"
            }
    except Exception as e:
        logging.error(f"Database health check error: {e}")
        health_status["database"] = {
            "status": "unhealthy",
            "response_time_ms": 0,
            "error": str(e)
        }
        health_status["rl_engine"] = {
            "status": "unhealthy",
            "total_experiences": 0,
            "response_time_ms": 0,
            "error": str(e)
        }
    
    # Determine overall health
    statuses = [
        health_status["database"]["status"],
        health_status["rl_engine"]["status"]
    ]
    
    if all(s == "healthy" for s in statuses):
        overall_status = "healthy"
    elif any(s == "unhealthy" for s in statuses):
        overall_status = "unhealthy"
    else:
        overall_status = "degraded"
    
    health_status["overall_status"] = overall_status
    
    return jsonify(health_status), 200

# ============================================================================
# BULK OPERATIONS ENDPOINTS
# ============================================================================

@app.route('/api/admin/bulk/extend', methods=['POST'])
def admin_bulk_extend():
    """Extend licenses for multiple users"""
    api_key = request.headers.get('X-Admin-API-Key')
    if api_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    license_keys = data.get('license_keys', [])
    days = data.get('days', 30)
    
    if not license_keys:
        return jsonify({"error": "No license keys provided"}), 400
    
    if len(license_keys) > 100:
        return jsonify({"error": "Maximum 100 users per bulk operation"}), 400
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    success_count = 0
    failed_count = 0
    errors = []
    
    try:
        for key in license_keys:
            try:
                cur.execute("""
                    UPDATE users 
                    SET license_expiration = license_expiration + INTERVAL '%s days'
                    WHERE license_key = %s
                """, (days, key))
                if cur.rowcount > 0:
                    success_count += 1
                else:
                    failed_count += 1
                    errors.append(f"{key[:8]}... not found")
            except Exception as e:
                failed_count += 1
                errors.append(f"{key[:8]}...: {str(e)}")
        
        conn.commit()
        logging.info(f"Bulk extend: {success_count} succeeded, {failed_count} failed")
    except Exception as e:
        conn.rollback()
        logging.error(f"Bulk extend error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()
    
    return jsonify({
        "success": success_count,
        "failed": failed_count,
        "errors": errors[:10]  # Limit error list
    }), 200

@app.route('/api/admin/bulk/suspend', methods=['POST'])
def admin_bulk_suspend():
    """Suspend multiple user licenses"""
    api_key = request.headers.get('X-Admin-API-Key')
    if api_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    license_keys = data.get('license_keys', [])
    
    if not license_keys or len(license_keys) > 100:
        return jsonify({"error": "Invalid request"}), 400
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            UPDATE users 
            SET license_status = 'SUSPENDED'
            WHERE license_key = ANY(%s)
        """, (license_keys,))
        success_count = cur.rowcount
        conn.commit()
        logging.info(f"Bulk suspended {success_count} users")
        return jsonify({"success": success_count, "failed": 0, "errors": []}), 200
    except Exception as e:
        conn.rollback()
        logging.error(f"Bulk suspend error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@app.route('/api/admin/bulk/activate', methods=['POST'])
def admin_bulk_activate():
    """Activate multiple user licenses"""
    api_key = request.headers.get('X-Admin-API-Key')
    if api_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    license_keys = data.get('license_keys', [])
    
    if not license_keys or len(license_keys) > 100:
        return jsonify({"error": "Invalid request"}), 400
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            UPDATE users 
            SET license_status = 'ACTIVE'
            WHERE license_key = ANY(%s)
        """, (license_keys,))
        success_count = cur.rowcount
        conn.commit()
        logging.info(f"Bulk activated {success_count} users")
        return jsonify({"success": success_count, "failed": 0, "errors": []}), 200
    except Exception as e:
        conn.rollback()
        logging.error(f"Bulk activate error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

@app.route('/api/admin/bulk/delete', methods=['POST'])
def admin_bulk_delete():
    """Delete multiple user licenses"""
    api_key = request.headers.get('X-Admin-API-Key')
    if api_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    data = request.get_json()
    license_keys = data.get('license_keys', [])
    
    if not license_keys or len(license_keys) > 100:
        return jsonify({"error": "Invalid request"}), 400
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute("""
            DELETE FROM users 
            WHERE license_key = ANY(%s)
        """, (license_keys,))
        success_count = cur.rowcount
        conn.commit()
        logging.info(f"Bulk deleted {success_count} users")
        return jsonify({"success": success_count, "failed": 0, "errors": []}), 200
    except Exception as e:
        conn.rollback()
        logging.error(f"Bulk delete error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

# ============================================================================
# USER RETENTION METRICS ENDPOINT
# ============================================================================

@app.route('/api/admin/metrics/retention', methods=['GET'])
def admin_retention_metrics():
    """Get comprehensive retention and engagement metrics"""
    api_key = request.headers.get('X-Admin-API-Key')
    if api_key != ADMIN_API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Churn rate (last 30 days)
        cur.execute("""
            WITH expired_users AS (
                SELECT 
                    COUNT(*) as total_expired,
                    COUNT(*) FILTER (WHERE license_status = 'CANCELLED' OR license_status = 'EXPIRED') as churned
                FROM users
                WHERE license_expiration BETWEEN NOW() - INTERVAL '30 days' AND NOW()
            )
            SELECT 
                CASE WHEN total_expired > 0 
                    THEN (churned * 100.0 / total_expired)
                    ELSE 0 
                END as churn_rate
            FROM expired_users
        """)
        churn_data = cur.fetchone()
        churn_rate = float(churn_data['churn_rate']) if churn_data else 0.0
        
        # Average subscription length in months
        cur.execute("""
            SELECT 
                AVG(EXTRACT(DAY FROM license_expiration - created_at) / 30.0) as avg_months
            FROM users
            WHERE created_at IS NOT NULL
        """)
        avg_sub = cur.fetchone()
        avg_subscription_months = float(avg_sub['avg_months']) if avg_sub and avg_sub['avg_months'] else 0.0
        
        # Active usage rate (users with API calls in last 24h)
        cur.execute("""
            WITH active_licenses AS (
                SELECT COUNT(*) as total
                FROM users
                WHERE license_status = 'ACTIVE'
            ),
            recent_activity AS (
                SELECT COUNT(DISTINCT license_key) as active
                FROM api_logs
                WHERE timestamp >= NOW() - INTERVAL '24 hours'
            )
            SELECT 
                CASE WHEN al.total > 0
                    THEN (ra.active * 100.0 / al.total)
                    ELSE 0
                END as usage_rate
            FROM active_licenses al, recent_activity ra
        """)
        usage_data = cur.fetchone()
        active_usage_rate = float(usage_data['usage_rate']) if usage_data else 0.0
        
        # Renewal rate (users who renewed vs expired in last 30 days)
        cur.execute("""
            WITH expired_last_month AS (
                SELECT 
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE license_status = 'ACTIVE') as renewed
                FROM users
                WHERE license_expiration BETWEEN NOW() - INTERVAL '60 days' AND NOW() - INTERVAL '30 days'
            )
            SELECT 
                CASE WHEN total > 0
                    THEN (renewed * 100.0 / total)
                    ELSE 0
                END as renewal_rate
            FROM expired_last_month
        """)
        renewal_data = cur.fetchone()
        renewal_rate = float(renewal_data['renewal_rate']) if renewal_data else 0.0
        
        # Lifetime value
        cur.execute("""
            SELECT 
                AVG(
                    CASE 
                        WHEN license_type = 'MONTHLY' THEN (EXTRACT(DAY FROM license_expiration - created_at) / 30.0) * 200.00
                        WHEN license_type = 'ANNUAL' THEN (EXTRACT(DAY FROM license_expiration - created_at) / 365.0) * 2000.00
                        ELSE 0
                    END
                ) as avg_ltv
            FROM users
            WHERE created_at IS NOT NULL
        """)
        ltv_data = cur.fetchone()
        lifetime_value = float(ltv_data['avg_ltv']) if ltv_data and ltv_data['avg_ltv'] else 0.0
        
        # Inactive users (no API calls in 7+ days)
        cur.execute("""
            SELECT 
                l.account_id,
                l.email,
                MAX(a.timestamp) as last_active,
                EXTRACT(DAY FROM NOW() - MAX(a.timestamp)) as days_inactive
            FROM users l
            LEFT JOIN api_logs a ON l.license_key = a.license_key
            WHERE l.license_status = 'ACTIVE'
            GROUP BY l.account_id, l.email
            HAVING MAX(a.timestamp) < NOW() - INTERVAL '7 days' OR MAX(a.timestamp) IS NULL
            ORDER BY days_inactive DESC NULLS FIRST
            LIMIT 20
        """)
        inactive_users = cur.fetchall()
        
        # Cohort retention (last 12 months)
        cur.execute("""
            SELECT 
                TO_CHAR(DATE_TRUNC('month', created_at), 'YYYY-MM') as cohort_month,
                COUNT(*) as total_signups,
                COUNT(*) FILTER (WHERE license_status = 'ACTIVE') as still_active,
                CASE WHEN COUNT(*) > 0
                    THEN (COUNT(*) FILTER (WHERE license_status = 'ACTIVE') * 100.0 / COUNT(*))
                    ELSE 0
                END as retention_pct
            FROM users
            WHERE created_at >= NOW() - INTERVAL '12 months'
            GROUP BY DATE_TRUNC('month', created_at)
            ORDER BY cohort_month DESC
        """)
        cohort_retention = cur.fetchall()
        
        # Churn trend (this month vs last month)
        cur.execute("""
            SELECT 
                DATE_TRUNC('month', license_expiration) as month,
                COUNT(*) FILTER (WHERE license_status = 'CANCELLED' OR license_status = 'EXPIRED') * 100.0 / COUNT(*) as churn
            FROM users
            WHERE license_expiration >= NOW() - INTERVAL '60 days'
            GROUP BY DATE_TRUNC('month', license_expiration)
            ORDER BY month DESC
            LIMIT 2
        """)
        churn_trend_data = cur.fetchall()
        churn_trend = {
            "this_month": float(churn_trend_data[0]['churn']) if len(churn_trend_data) > 0 else churn_rate,
            "last_month": float(churn_trend_data[1]['churn']) if len(churn_trend_data) > 1 else churn_rate
        }
        
        return jsonify({
            "churn_rate": round(churn_rate, 2),
            "churn_trend": churn_trend,
            "avg_subscription_months": round(avg_subscription_months, 2),
            "active_usage_rate": round(active_usage_rate, 2),
            "renewal_rate": round(renewal_rate, 2),
            "lifetime_value": round(lifetime_value, 2),
            "inactive_users": [
                {
                    "account_id": user['account_id'][:12] + "..." if user['account_id'] else "N/A",
                    "email": user['email'],
                    "last_active": user['last_active'].isoformat() if user['last_active'] else "Never",
                    "days_inactive": int(user['days_inactive']) if user['days_inactive'] else 999
                }
                for user in inactive_users
            ],
            "cohort_retention": [
                {
                    "month": cohort['cohort_month'],
                    "signups": cohort['total_signups'],
                    "still_active": cohort['still_active'],
                    "retention": round(float(cohort['retention_pct']), 1)
                }
                for cohort in cohort_retention
            ]
        }), 200
    except Exception as e:
        logging.error(f"Retention metrics error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()

if __name__ == '__main__':
    init_database_if_needed()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

