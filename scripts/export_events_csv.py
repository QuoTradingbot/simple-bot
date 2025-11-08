#!/usr/bin/env python3
"""
Export Forex Factory Events to CSV

Exports the JSON calendar data to a CSV file for easy viewing in Excel/Sheets.
"""

import json
import csv
import os
from datetime import datetime

def export_to_csv():
    """Export events to CSV file."""
    # Load JSON data
    json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'forex_factory_events.json')
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # CSV output path
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'forex_factory_events.csv')
    
    # Write CSV
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['Date', 'Day', 'Time', 'Currency', 'Impact', 'Event', 'Forecast', 'Previous']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for event in data['events']:
            # Get day of week
            event_date = datetime.strptime(event['date'], '%Y-%m-%d')
            day_of_week = event_date.strftime('%A')
            
            writer.writerow({
                'Date': event['date'],
                'Day': day_of_week,
                'Time': event['time'],
                'Currency': event['currency'],
                'Impact': event['impact'].upper(),
                'Event': event['event'],
                'Forecast': event['forecast'],
                'Previous': event['previous']
            })
    
    print(f"✓ Exported {len(data['events'])} events to CSV")
    print(f"  File: {csv_path}")
    print(f"\n  Date range: {data['metadata']['date_range']['from']} to {data['metadata']['date_range']['to']}")
    
    # Print summary by impact
    high_count = sum(1 for e in data['events'] if e['impact'] == 'high')
    medium_count = sum(1 for e in data['events'] if e['impact'] == 'medium')
    low_count = sum(1 for e in data['events'] if e['impact'] == 'low')
    
    print(f"\n  Impact breakdown:")
    print(f"    High:   {high_count} events")
    print(f"    Medium: {medium_count} events")
    print(f"    Low:    {low_count} events")
    print(f"\n  You can now open the CSV file in Excel or Google Sheets!")

if __name__ == "__main__":
    export_to_csv()
