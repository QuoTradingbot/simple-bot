"""
Test script to send a sample license email
"""
import os
import sys
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

logging.basicConfig(level=logging.INFO)

# Email configuration
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "noreply@quotrading.com")

def send_test_email(to_email, license_key="TEST-DEMO-1234-5678"):
    """Send a test license email"""
    
    logging.info(f"🔍 Sending test email to {to_email}")
    logging.info(f"🔍 SENDGRID_API_KEY present: {bool(SENDGRID_API_KEY)}")
    logging.info(f"🔍 SMTP_USERNAME present: {bool(SMTP_USERNAME)}")
    logging.info(f"🔍 FROM_EMAIL: {FROM_EMAIL}")
    
    subject = "Your QuoTrading AI License Key"
    
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <h1 style="color: #667eea;">Welcome to QuoTrading AI</h1>
        
        <p>Thank you for your subscription! Your license key is below.</p>
        
        <div style="background: #f3f4f6; padding: 20px; border-radius: 10px; margin: 20px 0;">
            <h2 style="margin-top: 0;">Your License Key:</h2>
            <p style="font-size: 24px; font-weight: bold; color: #667eea; letter-spacing: 2px; font-family: monospace;">{license_key}</p>
            <p style="font-size: 14px; color: #6b7280; margin-top: 10px;">⚠️ Save this key securely - you'll need it to activate the AI</p>
        </div>
        
        <h2>Getting Started:</h2>
        <ol>
            <li><strong>Download the AI</strong> from your Whop dashboard or email</li>
            <li><strong>Launch the application</strong> on your computer</li>
            <li><strong>Enter your license key</strong> when prompted: <code style="background: #f3f4f6; padding: 2px 6px; border-radius: 4px; font-family: monospace;">{license_key}</code></li>
            <li><strong>Connect your brokerage account</strong> and set your trading preferences</li>
            <li><strong>Start trading</strong> with AI-powered market analysis</li>
        </ol>
        
        <h2>📌 Important:</h2>
        <ul>
            <li>🔐 Your license key is unique and should not be shared</li>
            <li>✅ This email contains everything you need to get started</li>
            <li>📥 Check your spam folder if you haven't received your download link</li>
            <li>💡 Join our community for tips, updates, and support</li>
        </ul>
        
        <h2>Need Help?</h2>
        <p>📧 Email: <a href="mailto:support@quotrading.com" style="color: #667eea;">support@quotrading.com</a></p>
        <p>💬 Discord: Join our community for live support</p>
        
        <p style="color: #6b7280; font-size: 12px; margin-top: 40px; border-top: 1px solid #e5e7eb; padding-top: 20px;">
            Your subscription renews monthly. Manage your subscription anytime from your Whop dashboard.
        </p>
    </body>
    </html>
    """
    
    try:
        # Try SendGrid first
        if SENDGRID_API_KEY:
            logging.info(f"🔍 Attempting SendGrid email to {to_email}")
            try:
                payload = {
                    "personalizations": [{"to": [{"email": to_email}]}],
                    "from": {"email": FROM_EMAIL, "name": "QuoTrading"},
                    "subject": subject,
                    "content": [{"type": "text/html", "value": html_body}]
                }
                
                response = requests.post(
                    "https://api.sendgrid.com/v3/mail/send",
                    headers={
                        "Authorization": f"Bearer {SENDGRID_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=payload,
                    timeout=10
                )
                logging.info(f"🔍 SendGrid response: status={response.status_code}, body={response.text}")
                
                if response.status_code == 202:
                    logging.info(f"✅ SendGrid email sent successfully to {to_email}")
                    return True
                else:
                    logging.error(f"❌ SendGrid failed: {response.status_code} - {response.text}")
                    logging.warning(f"Trying SMTP fallback")
            except Exception as e:
                logging.error(f"❌ SendGrid exception: {type(e).__name__}: {str(e)}")
                logging.warning(f"Trying SMTP fallback")
        else:
            logging.warning(f"⚠️ SENDGRID_API_KEY not configured")
        
        # Fallback to SMTP
        if SMTP_USERNAME and SMTP_PASSWORD:
            logging.info(f"🔍 Attempting SMTP email to {to_email}")
            try:
                msg = MIMEMultipart('alternative')
                msg['Subject'] = subject
                msg['From'] = FROM_EMAIL
                msg['To'] = to_email
                msg.attach(MIMEText(html_body, 'html'))
                
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
                    server.starttls()
                    server.login(SMTP_USERNAME, SMTP_PASSWORD)
                    server.send_message(msg)
                logging.info(f"✅ SMTP email sent successfully to {to_email}")
                return True
            except Exception as e:
                logging.error(f"❌ SMTP exception: {type(e).__name__}: {str(e)}")
                logging.error(f"❌ No email method worked")
                return False
        else:
            logging.error(f"❌ No email method configured - SMTP credentials missing")
            return False
            
    except Exception as e:
        logging.error(f"❌ CRITICAL ERROR: {type(e).__name__}: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_email.py <your-email@example.com>")
        print("Example: python test_email.py kevin@example.com")
        sys.exit(1)
    
    test_email = sys.argv[1]
    print(f"\n{'='*60}")
    print(f"📧 Sending test QuoTrading license email")
    print(f"{'='*60}")
    print(f"To: {test_email}")
    print(f"From: {FROM_EMAIL}")
    print(f"License Key: TEST-DEMO-1234-5678")
    print(f"{'='*60}\n")
    
    success = send_test_email(test_email)
    
    if success:
        print(f"\n✅ SUCCESS! Check your inbox at {test_email}")
        print(f"📬 Don't forget to check spam/junk folder if you don't see it")
    else:
        print(f"\n❌ FAILED to send email")
        print(f"Check the logs above for details")
        print(f"\nMake sure you have either:")
        print(f"  1. SENDGRID_API_KEY environment variable set, OR")
        print(f"  2. SMTP_USERNAME and SMTP_PASSWORD environment variables set")
