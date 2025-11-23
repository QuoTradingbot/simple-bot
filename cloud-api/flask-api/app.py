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
    """Validate license key against PostgreSQL database"""
    if not license_key:
        return False, "License key required"
    
    conn = get_db_connection()
    if not conn:
        return False, "Database connection failed"
    
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
                return False, "Invalid license key"
            
            # Check if license is active (case-insensitive)
            if user['license_status'].lower() != 'active':
                return False, f"License is {user['license_status']}"
            
            # Check expiration
            if user['license_expiration']:
                if datetime.now() > user['license_expiration']:
                    return False, "License expired"
            
            # Log successful validation
            cursor.execute("""
                INSERT INTO api_logs (license_key, endpoint, request_data, status_code)
                VALUES (%s, %s, %s, %s)
            """, (license_key, '/api/main', '{"action": "validate"}', 200))
            conn.commit()
            
            return True, f"Valid {user['license_type']} license"
            
    except Exception as e:
        logging.error(f"License validation error: {e}")
        return False, str(e)
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
        is_valid, message = validate_license(license_key)
        
        if not is_valid:
            return jsonify({
                "status": "error",
                "message": message,
                "license_valid": False
            }), 403
        
        # Process signal with RL brain
        signal_type = data.get('signal_type', 'NEUTRAL')
        regime = data.get('regime', 'RANGING')
        vix_level = data.get('vix_level', 15.0)
        
        experiences = load_experiences()
        confidence = calculate_confidence(signal_type, regime, vix_level, experiences)
        
        response = {
            "status": "success",
            "license_valid": True,
            "message": message,
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
                FROM licenses 
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
                FROM licenses 
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

if __name__ == '__main__':
    init_database_if_needed()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

