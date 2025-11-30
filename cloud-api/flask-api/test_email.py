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
    
    subject = "🚀 Your QuoTrading AI License Key"
    
    # Test with sample Whop IDs
    whop_user_id = "user_abc123xyz"
    whop_membership_id = "mem_test123456"
    
    whop_id_html = f"""<p style="color: #334155; font-size: 14px; line-height: 1.6; margin: 0;">
                                <strong>Whop ID:</strong> <a href="https://whop.com" style="color: #667eea; text-decoration: none;">{whop_user_id}</a>
                            </p>"""
    
    order_link_html = f"""
                            <div style="background: #f8fafc; padding: 20px; border-radius: 8px; margin: 24px 0 0 0; text-align: center;">
                                <p style="color: #64748b; font-size: 14px; line-height: 1.6; margin: 0 0 12px 0;">
                                    <strong>Order Details</strong>
                                </p>
                                <p style="color: #334155; font-size: 13px; line-height: 1.6; margin: 0 0 12px 0;">
                                    Invoice: R-{whop_membership_id[-8:]}
                                </p>
                                <p style="margin: 0;">
                                    <a href="https://whop.com/hub/memberships/{whop_membership_id}" style="display: inline-block; background: #667eea; color: #ffffff; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; margin-right: 8px;">Access Order</a>
                                    <a href="https://whop.com/hub/memberships/{whop_membership_id}/invoice" style="display: inline-block; background: #ffffff; color: #667eea; padding: 10px 20px; border-radius: 6px; text-decoration: none; font-size: 14px; font-weight: 600; border: 2px solid #667eea;">View Invoice</a>
                                </p>
                            </div>
            """
    
    html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin: 0; padding: 0; background-color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f8fafc; padding: 40px 20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 16px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07); overflow: hidden;">
                    
                    <!-- Header -->
                    <tr>
                        <td style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; text-align: center;">
                            <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 700; letter-spacing: -0.5px;">
                                Welcome to QuoTrading AI
                            </h1>
                            <p style="color: rgba(255, 255, 255, 0.9); margin: 10px 0 0 0; font-size: 16px;">
                                Your AI-powered trading journey starts now
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Whop ID Section -->
                    <tr>
                        <td style="background: #f8fafc; padding: 20px 40px; border-bottom: 1px solid #e2e8f0;">
                            {whop_id_html}
                        </td>
                    </tr>
                    
                    <!-- License Key Box -->
                    <tr>
                        <td style="padding: 40px;">
                            <p style="color: #334155; font-size: 16px; line-height: 1.6; margin: 0 0 24px 0;">
                                Thank you for subscribing! Your license key is unique to your account — do not share it. Save this email for future reference.
                            </p>
                            
                            <div style="background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%); border-left: 4px solid #667eea; padding: 24px; border-radius: 8px; margin: 24px 0;">
                                <p style="color: #64748b; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; margin: 0 0 8px 0;">
                                    Your License Key
                                </p>
                                <p style="font-size: 28px; font-weight: 700; color: #1e293b; letter-spacing: 1px; font-family: 'Courier New', monospace; margin: 0; word-break: break-all;">
                                    {license_key}
                                </p>
                                <p style="color: #f59e0b; font-size: 13px; margin: 12px 0 0 0; font-weight: 500;">
                                    ⚠️ Keep this key secure — it's unique to your account
                                </p>
                            </div>
                            
                            <!-- Getting Started -->
                            <h2 style="color: #1e293b; font-size: 20px; font-weight: 700; margin: 32px 0 16px 0;">
                                Getting Started
                            </h2>
                            
                            <table width="100%" cellpadding="0" cellspacing="0" style="margin: 0 0 24px 0;">
                                <tr>
                                    <td style="padding: 8px 0;">
                                        <p style="color: #334155; font-size: 15px; line-height: 1.6; margin: 0;">
                                            <strong style="color: #667eea;">1.</strong> <strong>Download the AI</strong> — Check your email for the download link
                                        </p>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0;">
                                        <p style="color: #334155; font-size: 15px; line-height: 1.6; margin: 0;">
                                            <strong style="color: #667eea;">2.</strong> <strong>Launch the application</strong> — Run the QuoTrading AI on your computer
                                        </p>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0;">
                                        <p style="color: #334155; font-size: 15px; line-height: 1.6; margin: 0;">
                                            <strong style="color: #667eea;">3.</strong> <strong>Enter your license key</strong> — Paste the key above when prompted
                                        </p>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0;">
                                        <p style="color: #334155; font-size: 15px; line-height: 1.6; margin: 0;">
                                            <strong style="color: #667eea;">4.</strong> <strong>Connect your broker</strong> — Enter your brokerage API credentials (username and API key)
                                        </p>
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 8px 0;">
                                        <p style="color: #334155; font-size: 15px; line-height: 1.6; margin: 0;">
                                            <strong style="color: #667eea;">5.</strong> <strong>Start trading</strong> — Begin using AI-powered market analysis
                                        </p>
                                    </td>
                                </tr>
                            </table>
                            
                            <!-- Important Notes -->
                            <div style="background: #fef3c7; border-left: 4px solid #f59e0b; padding: 16px 20px; border-radius: 8px; margin: 24px 0 16px 0;">
                                <p style="color: #92400e; font-size: 14px; font-weight: 600; margin: 0 0 12px 0;">
                                    ⚠️ Important Information
                                </p>
                                <p style="color: #78350f; font-size: 14px; line-height: 1.8; margin: 0;">
                                    • You'll need API credentials from your broker to connect (contact your broker for API access)
                                </p>
                            </div>
                        </td>
                    </tr>
                    
                    <!-- Order Details Section -->
                    <tr>
                        <td style="padding: 0 40px 24px 40px;">
                            {order_link_html}
                        </td>
                    </tr>
                    
                    <!-- Support Section -->
                    <tr>
                        <td style="padding: 0 40px 40px 40px;">
                            <h2 style="color: #1e293b; font-size: 20px; font-weight: 700; margin: 0 0 16px 0;">
                                Need Help?
                            </h2>
                            <p style="color: #334155; font-size: 15px; line-height: 1.6; margin: 0 0 12px 0;">
                                <strong>📧 Email Support:</strong>
                                <a href="mailto:support@quotrading.com" style="color: #667eea; text-decoration: none;">support@quotrading.com</a>
                            </p>
                            <p style="color: #334155; font-size: 15px; line-height: 1.6; margin: 0;">
                                <strong>💬 Discord Community:</strong> Get live support and connect with other traders
                            </p>
                        </td>
                    </tr>
                    
                    <!-- Footer -->
                    <tr>
                        <td style="background: #f8fafc; padding: 32px; text-align: center; border-top: 1px solid #e2e8f0;">
                            <p style="color: #64748b; font-size: 13px; line-height: 1.6; margin: 0 0 8px 0;">
                                Your subscription renews monthly and can be managed anytime from your Whop dashboard.
                            </p>
                            <p style="color: #94a3b8; font-size: 12px; margin: 0;">
                                © 2025 QuoTrading. All rights reserved.
                            </p>
                        </td>
                    </tr>
                    
                </table>
            </td>
        </tr>
    </table>
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
