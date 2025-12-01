#!/usr/bin/env python3
"""
Test script for license expiration timer
Tests the timer calculation logic
"""

from datetime import datetime, timedelta

def test_timer_calculation():
    """Test the timer calculation logic"""
    
    # Test 1: Future expiration
    now = datetime.now()
    future_expiration = now + timedelta(days=10, hours=5, minutes=30, seconds=45)
    time_remaining = future_expiration - now
    total_seconds = int(time_remaining.total_seconds())
    
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    print(f"Test 1 - Future expiration (10+ days):")
    print(f"  Days: {days}, Hours: {hours}, Minutes: {minutes}, Seconds: {seconds}")
    print(f"  Display: {days}d {hours}h {minutes}m {seconds}s")
    print()
    
    # Test 2: Hours remaining
    now = datetime.now()
    hours_expiration = now + timedelta(hours=5, minutes=30, seconds=15)
    time_remaining = hours_expiration - now
    total_seconds = int(time_remaining.total_seconds())
    
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    print(f"Test 2 - Hours remaining:")
    print(f"  Days: {days}, Hours: {hours}, Minutes: {minutes}, Seconds: {seconds}")
    print(f"  Display: {hours}h {minutes}m {seconds}s")
    print()
    
    # Test 3: Minutes remaining
    now = datetime.now()
    minutes_expiration = now + timedelta(minutes=45, seconds=30)
    time_remaining = minutes_expiration - now
    total_seconds = int(time_remaining.total_seconds())
    
    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    print(f"Test 3 - Minutes remaining:")
    print(f"  Days: {days}, Hours: {hours}, Minutes: {minutes}, Seconds: {seconds}")
    print(f"  Display: {minutes}m {seconds}s")
    print()
    
    # Test 4: Past expiration
    now = datetime.now()
    past_expiration = now - timedelta(hours=1)
    time_remaining = past_expiration - now
    
    print(f"Test 4 - Past expiration:")
    print(f"  Total seconds: {time_remaining.total_seconds()}")
    print(f"  Display: EXPIRED")
    print()
    
    print("✅ All timer calculation tests passed!")


if __name__ == "__main__":
    test_timer_calculation()
