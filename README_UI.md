# SmartAPI Candle Chart UI

Interactive web-based UI to view candlestick charts for different exchanges using SmartAPI.

## Features

- 📊 **Exchange Selection**: Choose from NSE, BSE, NFO, MCX
- 📈 **Interactive Candlestick Charts**: Beautiful, interactive charts with volume
- ⏱️ **Multiple Time Intervals**: ONE_MINUTE, FIVE_MINUTE, FIFTEEN_MINUTE, THIRTY_MINUTE, ONE_HOUR, ONE_DAY
- 📅 **Custom Date Range**: Select any date range for historical data
- 📥 **Data Export**: Download data as CSV
- 🔍 **Real-time Data**: Fetch live historical data from SmartAPI

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the UI

### Option 1: Using the batch file (Windows)
```bash
run_ui.bat
```

### Option 2: Using Streamlit command
```bash
streamlit run candle_chart_ui.py
```

The UI will open automatically in your default web browser at `http://localhost:8501`

## Usage

1. **Connect to SmartAPI**: 
   - Click "Connect to SmartAPI" in the sidebar
   - Make sure your `.env` file has the correct credentials

2. **Select Exchange**: 
   - Choose the exchange (NSE, BSE, NFO, MCX) from the dropdown

3. **Enter Symbol Details**:
   - Enter the Symbol Token (e.g., 3045 for SBIN)
   - Enter Symbol Name for display (e.g., SBIN)

4. **Select Time Interval**:
   - Choose the candle interval (ONE_MINUTE, FIVE_MINUTE, etc.)

5. **Set Date Range**:
   - Select From Date and Time
   - Select To Date and Time

6. **Fetch Data**:
   - Click "Fetch Data" button
   - The candlestick chart will be displayed

7. **View and Download**:
   - View the interactive chart
   - Expand "View Raw Data" to see the data table
   - Download data as CSV if needed

## Chart Features

- **Candlestick Chart**: Shows Open, High, Low, Close prices
- **Volume Chart**: Shows trading volume below the price chart
- **Interactive**: Zoom, pan, and hover for detailed information
- **Dark Theme**: Easy on the eyes

## Example Symbol Tokens

- **SBIN (State Bank of India)**: 3045 (NSE)
- **RELIANCE**: 2885 (NSE)
- **TCS**: 11536 (NSE)
- **INFY**: 4085 (NSE)

## Notes

- Make sure you're connected to SmartAPI before fetching data
- The date range should be within market hours for better results
- Large date ranges with small intervals may take longer to fetch
- Data is fetched in real-time from SmartAPI

## Troubleshooting

- **Connection Failed**: Check your `.env` file credentials
- **No Data**: Verify symbol token and date range
- **Chart Not Displaying**: Check if data was fetched successfully
- **Port Already in Use**: Change the port: `streamlit run candle_chart_ui.py --server.port 8502`



