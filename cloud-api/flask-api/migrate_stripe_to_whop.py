import os
import psycopg2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database configuration
DB_HOST = os.environ.get("DB_HOST", "quotrading-db.postgres.database.azure.com")
DB_NAME = os.environ.get("DB_NAME", "quotrading")
DB_USER = os.environ.get("DB_USER", "quotradingadmin")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_PORT = os.environ.get("DB_PORT", "5432")

def get_db_connection():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        logging.error(f"Error connecting to database: {e}")
        return None

def migrate_database():
    logging.info("Starting database migration: Stripe -> Whop")
    
    conn = get_db_connection()
    if not conn:
        logging.error("Could not connect to database. Aborting.")
        return

    try:
        with conn.cursor() as cursor:
            # 1. Add Whop columns
            logging.info("Adding Whop columns...")
            cursor.execute("""
                ALTER TABLE users 
                ADD COLUMN IF NOT EXISTS whop_membership_id VARCHAR(255),
                ADD COLUMN IF NOT EXISTS whop_user_id VARCHAR(255);
            """)
            
            # 2. Drop Stripe column (optional, maybe keep for backup?)
            # The user said "get rid of all strip logic n code", so let's drop it.
            logging.info("Dropping Stripe columns...")
            cursor.execute("""
                ALTER TABLE users 
                DROP COLUMN IF EXISTS stripe_customer_id;
            """)
            
            conn.commit()
            logging.info("✅ Database migration completed successfully!")
            
    except Exception as e:
        logging.error(f"Migration failed: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    if not DB_PASSWORD:
        logging.warning("⚠️ DB_PASSWORD environment variable not set. Migration might fail if not running in Azure.")
    
    migrate_database()
