#!/usr/bin/env python3
"""
Forex Factory Economic Calendar Scraper

This script scrapes upcoming economic events from Forex Factory's calendar
and saves them to a JSON file with dates, times, and event details.

NOTE: This script requires internet access to forexfactory.com. If running
in a restricted environment where forexfactory.com is blocked, a pre-populated
template file has been provided in data/forex_factory_events.json with typical
economic events.

The scraper fetches approximately 12 weeks (3 months) of future events including:
- Central bank decisions (FOMC, ECB, BoE, BoJ, RBA, RBNZ)
- Employment reports (NFP, Unemployment)
- Inflation data (CPI, PPI)
- Economic indicators (PMI, Retail Sales, GDP)
"""

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
import time
import sys
import os

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class ForexFactoryScraper:
    """Scrapes economic events from Forex Factory calendar."""
    
    def __init__(self):
        self.base_url = "https://www.forexfactory.com/calendar"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.events = []
        
    def get_calendar_page(self, date_str=None):
        """
        Fetch calendar page for a specific date.
        
        Args:
            date_str: Date in format 'mmddyyyy' or None for current week
            
        Returns:
            BeautifulSoup object of the page
        """
        url = self.base_url
        if date_str:
            url = f"{self.base_url}?day={date_str}"
            
        try:
            print(f"Fetching calendar for {date_str if date_str else 'current week'}...")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching calendar: {e}")
            return None
    
    def parse_event_row(self, row, current_date):
        """
        Parse a single event row from the calendar table.
        
        Args:
            row: BeautifulSoup tr element
            current_date: Current date being parsed
            
        Returns:
            Dictionary with event details or None
        """
        try:
            # Extract date if present
            date_cell = row.find('td', class_='calendar__cell calendar__date')
            if date_cell and date_cell.text.strip():
                date_text = date_cell.text.strip()
                # Parse date like "Fri Nov 8"
                try:
                    current_year = datetime.now().year
                    current_date = datetime.strptime(f"{date_text} {current_year}", "%a %b %d %Y").date()
                except:
                    pass
            
            # Extract time
            time_cell = row.find('td', class_='calendar__cell calendar__time')
            event_time = time_cell.text.strip() if time_cell else "All Day"
            
            # Extract currency
            currency_cell = row.find('td', class_='calendar__cell calendar__currency')
            currency = currency_cell.text.strip() if currency_cell else ""
            
            # Extract impact (low, medium, high)
            impact_cell = row.find('td', class_='calendar__cell calendar__impact')
            impact = "unknown"
            if impact_cell:
                impact_span = impact_cell.find('span')
                if impact_span:
                    impact_class = impact_span.get('class', [])
                    if 'icon--ff-impact-red' in impact_class:
                        impact = "high"
                    elif 'icon--ff-impact-ora' in impact_class:
                        impact = "medium"
                    elif 'icon--ff-impact-yel' in impact_class:
                        impact = "low"
            
            # Extract event name
            event_cell = row.find('td', class_='calendar__cell calendar__event')
            event_name = event_cell.text.strip() if event_cell else ""
            
            # Extract forecast
            forecast_cell = row.find('td', class_='calendar__cell calendar__forecast')
            forecast = forecast_cell.text.strip() if forecast_cell else ""
            
            # Extract previous value
            previous_cell = row.find('td', class_='calendar__cell calendar__previous')
            previous = previous_cell.text.strip() if previous_cell else ""
            
            # Only return if we have an event name
            if event_name and event_name != "":
                return {
                    'date': str(current_date),
                    'time': event_time,
                    'currency': currency,
                    'impact': impact,
                    'event': event_name,
                    'forecast': forecast,
                    'previous': previous
                }
            
        except Exception as e:
            print(f"Error parsing event row: {e}")
            
        return None
    
    def scrape_week(self, start_date):
        """
        Scrape events for a specific week.
        
        Args:
            start_date: datetime.date object for the start of the week
        """
        # Format date as mmddyyyy
        date_str = start_date.strftime("%m%d%Y")
        
        soup = self.get_calendar_page(date_str)
        if not soup:
            return
        
        # Find the calendar table
        calendar_table = soup.find('table', class_='calendar__table')
        if not calendar_table:
            print(f"No calendar table found for {date_str}")
            return
        
        # Parse each row
        current_date = start_date
        rows = calendar_table.find_all('tr', class_='calendar__row')
        
        for row in rows:
            event = self.parse_event_row(row, current_date)
            if event:
                self.events.append(event)
        
        print(f"Scraped {len([e for e in self.events if e['date'] == str(current_date)])} events for week of {date_str}")
        
        # Be nice to the server
        time.sleep(2)
    
    def scrape_future_events(self, weeks_ahead=12):
        """
        Scrape events for multiple weeks into the future.
        
        Args:
            weeks_ahead: Number of weeks to scrape ahead
        """
        print(f"Starting to scrape Forex Factory calendar for {weeks_ahead} weeks...")
        
        today = datetime.now().date()
        
        # Start from the beginning of current week (Monday)
        start_of_week = today - timedelta(days=today.weekday())
        
        for week in range(weeks_ahead):
            week_start = start_of_week + timedelta(weeks=week)
            self.scrape_week(week_start)
        
        print(f"\nTotal events scraped: {len(self.events)}")
    
    def save_to_json(self, filename):
        """Save scraped events to JSON file."""
        output_path = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
        
        # Create metadata
        data = {
            'metadata': {
                'scraped_at': datetime.now().isoformat(),
                'total_events': len(self.events),
                'date_range': {
                    'from': min([e['date'] for e in self.events]) if self.events else None,
                    'to': max([e['date'] for e in self.events]) if self.events else None
                }
            },
            'events': self.events
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nEvents saved to: {output_path}")
        return output_path


def main():
    """Main function to run the scraper."""
    scraper = ForexFactoryScraper()
    
    # Scrape 12 weeks ahead (about 3 months)
    scraper.scrape_future_events(weeks_ahead=12)
    
    # Save to JSON file
    output_file = scraper.save_to_json('forex_factory_events.json')
    
    print("\n" + "="*60)
    print("Scraping completed successfully!")
    print("="*60)
    print(f"Output file: {output_file}")
    print(f"Total events: {len(scraper.events)}")
    
    # Show sample of events
    if scraper.events:
        print("\nSample of upcoming high-impact events:")
        high_impact = [e for e in scraper.events if e['impact'] == 'high'][:10]
        for event in high_impact:
            print(f"  {event['date']} {event['time']} - {event['currency']} - {event['event']}")


if __name__ == "__main__":
    main()
