"""
Admin endpoint to trigger pre-compute job manually or via scheduler
"""

# Add this to app.py (will be added via deployment)

@app.route('/api/admin/precompute', methods=['POST'])
def admin_precompute():
    """
    Admin endpoint to trigger pre-compute job.
    Called nightly by Azure Logic App at 3 AM.
    """
    try:
        # Verify admin key
        admin_key = request.headers.get('X-Admin-Key')
        expected_key = os.environ.get('ADMIN_KEY', 'change-me-in-production')
        
        if admin_key != expected_key:
            return jsonify({"error": "Unauthorized"}), 401
        
        # Run pre-compute in background thread (don't block response)
        import threading
        from precompute_confidence import precompute_all_buckets
        
        def run_precompute():
            try:
                logging.info("🔄 Starting scheduled pre-compute job...")
                success = precompute_all_buckets()
                if success:
                    logging.info("✅ Scheduled pre-compute completed successfully")
                else:
                    logging.error("❌ Scheduled pre-compute failed")
            except Exception as e:
                logging.error(f"❌ Pre-compute error: {e}")
        
        thread = threading.Thread(target=run_precompute, daemon=True)
        thread.start()
        
        return jsonify({
            "status": "started",
            "message": "Pre-compute job started in background",
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        logging.error(f"Admin precompute error: {e}")
        return jsonify({"error": str(e)}), 500
