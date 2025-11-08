# Forex Factory Economic Calendar

This directory contains economic calendar data based on typical Forex Factory calendar scheduling patterns.

## ⚠️ Important Note on Dates

**The dates in this calendar follow standard economic release schedules:**
- **NFP (Non-Farm Payrolls)**: First Friday of each month at 8:30am ET
- **CPI (Consumer Price Index)**: Mid-month (typically 10th-15th) at 8:30am ET
- **FOMC Meetings**: Scheduled 8 times per year (approximately every 6 weeks)
- **ECB/BoE/BoJ**: Monthly or bi-monthly meetings on standard weekdays

These are **realistic scheduling patterns** but should be verified with the actual calendar closer to the event date, as dates can occasionally shift due to holidays or special circumstances.

## Files

- **`forex_factory_events.json`** - Main file containing upcoming economic events with dates, times, and impact levels
- **`forex_factory_events.csv`** - Same data in CSV format for spreadsheet analysis

## Data Structure

The JSON file contains:

```json
{
  "metadata": {
    "scraped_at": "ISO timestamp",
    "total_events": "number",
    "date_range": {
      "from": "YYYY-MM-DD",
      "to": "YYYY-MM-DD"
    }
  },
  "events": [
    {
      "date": "YYYY-MM-DD",
      "time": "HH:MMam/pm or All Day",
      "currency": "USD/EUR/GBP/JPY/etc",
      "impact": "high/medium/low",
      "event": "Event name",
      "forecast": "Forecasted value",
      "previous": "Previous value"
    }
  ]
}
```

## Impact Levels

- **High** (Red) - Major market-moving events:
  - Central bank interest rate decisions (FOMC, ECB, BoE, BoJ, RBA, RBNZ)
  - Non-Farm Payrolls (NFP)
  - Consumer Price Index (CPI)
  - Producer Price Index (PPI)
  - GDP releases
  - Employment data

- **Medium** (Orange) - Moderate impact events:
  - PMI (Purchasing Managers Index)
  - Retail Sales
  - Industrial Production
  - Consumer Confidence
  - Trade Balance

- **Low** (Yellow) - Minor impact events:
  - Market holidays
  - Minor speeches
  - Low-impact indicators

## Key Events to Watch

### Monthly Recurring Events

**First Friday of Month (8:30am EST)**
- U.S. Nonfarm Payrolls (NFP)
- U.S. Unemployment Rate
- Average Hourly Earnings

**Mid-Month**
- U.S. CPI (Consumer Price Index) - ~13th
- U.S. PPI (Producer Price Index) - ~14th
- U.S. Retail Sales - ~15th

**Every 6 Weeks**
- FOMC Interest Rate Decision
- FOMC Press Conference (Fed Chair Powell)

**Monthly**
- ECB Interest Rate Decision (Usually 2nd or 3rd Thursday)
- BoE Interest Rate Decision (Usually 3rd Thursday)
- BoJ Interest Rate Decision (Usually 3rd Friday)

### Weekly Events

**Every Thursday (8:30am EST)**
- Unemployment Claims

**Last Friday of Month**
- PMI Flash releases (Manufacturing & Services)

## Using This Data

This economic calendar data can be used to:

1. **Avoid Trading** during high-impact events if you don't want volatility
2. **Plan Trades** around major economic releases
3. **Set Alerts** for important upcoming events
4. **Backtest Strategy** by avoiding or targeting news events
5. **Adjust Position Sizing** ahead of volatile periods

## Updating the Data

To update with fresh data from Forex Factory, run:

```bash
python scripts/scrape_forex_factory.py
```

This will fetch the latest events and update `forex_factory_events.json`.

**Note:** The scraper fetches approximately 12 weeks (3 months) of future events.

## Time Zones

All times are in Eastern Time (ET) unless otherwise specified. Forex Factory uses ET as the standard timezone.

## Integration with Trading Bot

The trading bot can be configured to:

- Pause trading 15-30 minutes before high-impact events
- Resume trading 15-30 minutes after events
- Adjust position sizes during event-heavy days
- Close positions before major central bank decisions

## Sources

Data source: [Forex Factory Economic Calendar](https://www.forexfactory.com/calendar)

---

**Last Updated:** 2025-11-08
