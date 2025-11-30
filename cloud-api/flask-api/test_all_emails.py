"""
Test script for all QuoTrading email types
"""
import os
import sys
from datetime import datetime, timedelta

# Import email functions from app.py
sys.path.insert(0, os.path.dirname(__file__))
from app import (
    send_license_email,
    send_renewal_email,
    send_cancellation_email,
    send_payment_failed_email,
    send_subscription_expired_email
)

def test_all_emails(email):
    """Test all email templates"""
    print("=" * 60)
    print("📧 Testing All QuoTrading Email Templates")
    print("=" * 60)
    print(f"Recipient: {email}")
    print("=" * 60)
    
    # Test data
    license_key = "TEST-DEMO-1234-5678"
    whop_user_id = "user_abc123xyz"
    whop_membership_id = "mem_test123456"
    
    today = datetime.now().strftime("%B %d, %Y")
    future_30 = (datetime.now() + timedelta(days=30)).strftime("%B %d, %Y")
    future_3 = (datetime.now() + timedelta(days=3)).strftime("%B %d, %Y")
    
    emails_to_send = [
        ("License Purchase", lambda: send_license_email(email, license_key, whop_user_id, whop_membership_id)),
        ("Subscription Renewal", lambda: send_renewal_email(email, today, future_30, whop_membership_id)),
        ("Cancellation Confirmation", lambda: send_cancellation_email(email, today, future_30, whop_membership_id)),
        ("Payment Failed", lambda: send_payment_failed_email(email, future_3, whop_membership_id)),
        ("Subscription Expired", lambda: send_subscription_expired_email(email, today))
    ]
    
    for email_type, send_func in emails_to_send:
        print(f"\n📨 Sending: {email_type}")
        print("-" * 60)
        try:
            success = send_func()
            if success:
                print(f"✅ {email_type} email sent successfully!")
            else:
                print(f"❌ {email_type} email failed to send")
        except Exception as e:
            print(f"❌ Error sending {email_type}: {e}")
        print("-" * 60)
    
    print("\n" + "=" * 60)
    print("✅ All email tests completed!")
    print(f"📬 Check your inbox at {email}")
    print("=" * 60)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        recipient_email = sys.argv[1]
    else:
        recipient_email = "kevinsuero072897@gmail.com"
    
    test_all_emails(recipient_email)
