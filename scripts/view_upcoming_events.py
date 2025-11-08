#!/usr/bin/env python3
"""
View Upcoming Forex Factory Economic Events

Quick utility to view upcoming high-impact economic events from the calendar.
"""

import json
import os
from datetime import datetime, timedelta

def load_events():
    """Load events from the JSON file."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'forex_factory_events.json')
    with open(data_path, 'r') as f:
        return json.load(f)

def get_upcoming_events(days_ahead=7, min_impact='medium'):
    """
    Get upcoming events within specified days.
    
    Args:
        days_ahead: Number of days to look ahead
        min_impact: Minimum impact level ('high', 'medium', 'low')
    
    Returns:
        List of events
    """
    data = load_events()
    today = datetime.now().date()
    end_date = today + timedelta(days=days_ahead)
    
    # Impact hierarchy
    impact_levels = {'low': 0, 'medium': 1, 'high': 2}
    min_level = impact_levels.get(min_impact, 1)
    
    upcoming = []
    for event in data['events']:
        event_date = datetime.strptime(event['date'], '%Y-%m-%d').date()
        event_impact_level = impact_levels.get(event['impact'], 0)
        
        if today <= event_date <= end_date and event_impact_level >= min_level:
            upcoming.append(event)
    
    return upcoming

def print_events(events, title="Upcoming Events"):
    """Print events in a formatted table."""
    if not events:
        print(f"\n{title}: None found")
        return
    
    print(f"\n{title}")
    print("=" * 80)
    print(f"{'Date':<12} {'Time':<10} {'Curr':<5} {'Impact':<8} {'Event':<45}")
    print("-" * 80)
    
    for event in events:
        impact_symbol = {
            'high': '🔴',
            'medium': '🟠', 
            'low': '🟡'
        }.get(event['impact'], '  ')
        
        print(f"{event['date']:<12} {event['time']:<10} {event['currency']:<5} "
              f"{impact_symbol} {event['impact']:<6} {event['event'][:44]:<45}")
    
    print("=" * 80)
    print(f"Total: {len(events)} events")

def main():
    """Main function."""
    print("\n" + "=" * 80)
    print(" " * 20 + "FOREX FACTORY ECONOMIC CALENDAR")
    print("=" * 80)
    
    data = load_events()
    metadata = data['metadata']
    
    print(f"\nCalendar Info:")
    print(f"  Total Events: {metadata['total_events']}")
    print(f"  Date Range: {metadata['date_range']['from']} to {metadata['date_range']['to']}")
    
    # Show this week's high-impact events
    this_week = get_upcoming_events(days_ahead=7, min_impact='high')
    print_events(this_week, "This Week's HIGH IMPACT Events")
    
    # Show next 2 weeks medium+ impact
    next_2_weeks = get_upcoming_events(days_ahead=14, min_impact='medium')
    print_events(next_2_weeks, "Next 2 Weeks - MEDIUM+ Impact Events")
    
    print("\n💡 Tips:")
    print("  🔴 High Impact = Major market movers (FOMC, NFP, CPI, etc.)")
    print("  🟠 Medium Impact = Moderate volatility (PMI, Retail Sales, etc.)")
    print("  🟡 Low Impact = Minor events (Holidays, speeches, etc.)")
    print("\n  Avoid trading 15-30 minutes before/after high-impact events!")
    print()

if __name__ == "__main__":
    main()
