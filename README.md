# SmartAPI-Python - Historical Data Fetcher

Python script to fetch historical candle data from Angel's Trading platform using SmartAPI-Python library.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with pycrypto, run:
```bash
pip uninstall pycrypto
pip install pycryptodome
```

## Configuration

### Option 1: Environment Variables (Recommended)

Set the following environment variables:

```bash
# Windows PowerShell
$env:SMARTAPI_KEY="Your Api Key"
$env:SMARTAPI_USERNAME="Your client code"
$env:SMARTAPI_PASSWORD="Your pin"
$env:SMARTAPI_TOTP_TOKEN="Your QR value"

# Windows CMD
set SMARTAPI_KEY=Your Api Key
set SMARTAPI_USERNAME=Your client code
set SMARTAPI_PASSWORD=Your pin
set SMARTAPI_TOTP_TOKEN=Your QR value

# Linux/Mac
export SMARTAPI_KEY="Your Api Key"
export SMARTAPI_USERNAME="Your client code"
export SMARTAPI_PASSWORD="Your pin"
export SMARTAPI_TOTP_TOKEN="Your QR value"
```

### Option 2: Edit the Script

Edit the `main()` function in `fetch_historical_data.py` and update:
- `API_KEY`: Your API key from Angel One
- `USERNAME`: Your client code
- `PASSWORD`: Your pin/password
- `TOTP_TOKEN`: Your TOTP token (QR value) for 2FA

## Usage

### Basic Usage

```python
from fetch_historical_data import SmartAPIHistoricalData

# Initialize the API client
api = SmartAPIHistoricalData(
    api_key="Your Api Key",
    username="Your client code",
    password="Your pin",
    totp_token="Your QR value"
)

# Connect and authenticate
api.connect()

# Fetch historical candle data
data = api.get_historical_candle_data(
    exchange="NSE",
    symboltoken="3045",  # SBIN token
    interval="ONE_MINUTE",
    fromdate="2024-01-01 09:00",
    todate="2024-01-01 16:00"
)

# Save to file
api.save_to_file(data, "output.json")

# Disconnect
api.disconnect()
```

### Run the Script

```bash
python fetch_historical_data.py
```

## Parameters

### Exchange
- `"NSE"` - National Stock Exchange
- `"BSE"` - Bombay Stock Exchange
- `"NFO"` - NSE Futures & Options
- `"MCX"` - Multi Commodity Exchange

### Interval Options
- `"ONE_MINUTE"` - 1 minute candles
- `"FIVE_MINUTE"` - 5 minute candles
- `"FIFTEEN_MINUTE"` - 15 minute candles
- `"THIRTY_MINUTE"` - 30 minute candles
- `"ONE_HOUR"` - 1 hour candles
- `"ONE_DAY"` - Daily candles

### Date Format
- Format: `"YYYY-MM-DD HH:MM"`
- Example: `"2024-01-15 09:00"`

## API Methods

- `connect()`: Establish connection and authenticate with the API
- `get_historical_candle_data(exchange, symboltoken, interval, fromdate, todate)`: Fetch historical candle data
- `get_profile()`: Fetch user profile information
- `save_to_file(data, filename)`: Save fetched data to a JSON file
- `disconnect()`: Terminate the session and logout

## Example Output

The script will fetch historical candle data and save it to a JSON file. Each candle contains:
- Open price
- High price
- Low price
- Close price
- Volume
- Timestamp

## Notes

- The TOTP token is generated automatically from your QR value
- Make sure your API credentials are valid and not expired
- The script includes error handling and logging
- All responses are returned as Python dictionaries
- Data is automatically saved to JSON files for later analysis

## Getting Your Credentials

1. **API Key**: Get from Angel One developer portal
2. **Client Code**: Your Angel One client ID
3. **Password**: Your trading pin
4. **TOTP Token**: Scan QR code from Angel One app to get the token

## Troubleshooting

- **Authentication Failed**: Check your credentials and TOTP token
- **Invalid Symbol Token**: Verify the symbol token for your stock
- **Date Range Error**: Ensure fromdate is before todate
- **Network Error**: Check your internet connection
