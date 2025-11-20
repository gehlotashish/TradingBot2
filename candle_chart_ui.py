"""
SmartAPI Candle Chart UI
Interactive UI to select exchange and view candlestick charts
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from SmartApi import SmartConnect
from SmartApi.smartWebSocketV2 import SmartWebSocketV2
import pyotp
from logzero import logger
import json
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pandas as pd
import threading
import time

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="SmartAPI Candle Chart",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'smartApi' not in st.session_state:
    st.session_state.smartApi = None
if 'authToken' not in st.session_state:
    st.session_state.authToken = None
if 'connected' not in st.session_state:
    st.session_state.connected = False
if 'liveApi' not in st.session_state:
    st.session_state.liveApi = None
if 'liveConnected' not in st.session_state:
    st.session_state.liveConnected = False
if 'liveTicks' not in st.session_state:
    st.session_state.liveTicks = []
if 'selectedSymbolForLive' not in st.session_state:
    st.session_state.selectedSymbolForLive = None
if 'ws' not in st.session_state:
    st.session_state.ws = None
if 'wsThread' not in st.session_state:
    st.session_state.wsThread = None
if 'ws_error' not in st.session_state:
    st.session_state.ws_error = None
if 'subscription_feed' not in st.session_state:
    st.session_state.subscription_feed = None
if 'ws_messages' not in st.session_state:
    st.session_state.ws_messages = []
if 'message_count' not in st.session_state:
    st.session_state.message_count = 0
if 'last_message_time' not in st.session_state:
    st.session_state.last_message_time = None
if 'tick_count' not in st.session_state:
    st.session_state.tick_count = 0


def connect_to_api():
    """Connect to SmartAPI for historical data"""
    try:
        API_KEY = os.getenv('HISTORICAL_DATA_API_KEY', os.getenv('API_KEY', '')).strip()
        USERNAME = os.getenv('CLIENT_CODE', '').strip()
        PASSWORD = os.getenv('PIN', '').strip()
        TOTP_TOKEN = os.getenv('TOTP_SECRET', '').strip()
        
        if not all([API_KEY, USERNAME, PASSWORD, TOTP_TOKEN]):
            return False, "Missing credentials in .env file"
        
        smartApi = SmartConnect(API_KEY)
        totp = pyotp.TOTP(TOTP_TOKEN).now()
        correlation_id = "abcde"
        data = smartApi.generateSession(USERNAME, PASSWORD, totp)
        
        if data['status'] == False:
            return False, f"Authentication failed: {data}"
        
        authToken = data['data']['jwtToken']
        refreshToken = data['data']['refreshToken']
        feedToken = smartApi.getfeedToken()
        
        st.session_state.smartApi = smartApi
        st.session_state.authToken = authToken
        st.session_state.refreshToken = refreshToken
        st.session_state.feedToken = feedToken
        st.session_state.connected = True
        
        return True, "Connected successfully"
    except Exception as e:
        logger.exception(f"Connection error: {e}")
        return False, str(e)


def connect_to_live_api():
    """Connect to SmartAPI for live data"""
    try:
        LIVE_API_KEY = os.getenv('LIVE_DATA_API_KEY', '').strip()
        USERNAME = os.getenv('CLIENT_CODE', '').strip()
        PASSWORD = os.getenv('PIN', '').strip()
        TOTP_TOKEN = os.getenv('TOTP_SECRET', '').strip()
        
        if not all([LIVE_API_KEY, USERNAME, PASSWORD, TOTP_TOKEN]):
            return False, "Missing LIVE_DATA_API_KEY in .env file"
        
        liveApi = SmartConnect(LIVE_API_KEY)
        totp = pyotp.TOTP(TOTP_TOKEN).now()
        data = liveApi.generateSession(USERNAME, PASSWORD, totp)
        
        if data['status'] == False:
            return False, f"Live API Authentication failed: {data}"
        
        authToken = data['data']['jwtToken']
        feedToken = liveApi.getfeedToken()
        
        st.session_state.liveApi = liveApi
        st.session_state.liveAuthToken = authToken
        st.session_state.liveFeedToken = feedToken
        st.session_state.liveConnected = True
        
        return True, "Live data connected successfully"
    except Exception as e:
        logger.exception(f"Live connection error: {e}")
        return False, str(e)


def on_ticks(ws, message):
    """Callback function for live tick data"""
    try:
        current_time = datetime.now()
        logger.debug(f"Received message: {message}")
        
        # Update message tracking
        if 'message_count' not in st.session_state:
            st.session_state.message_count = 0
        if 'last_message_time' not in st.session_state:
            st.session_state.last_message_time = None
        
        st.session_state.message_count += 1
        st.session_state.last_message_time = current_time
        
        # Store last few messages for debugging
        if 'ws_messages' not in st.session_state:
            st.session_state.ws_messages = []
        st.session_state.ws_messages.insert(0, {
            'timestamp': current_time.strftime("%H:%M:%S.%f")[:-3],
            'message': str(message)[:200]  # Truncate long messages
        })
        if len(st.session_state.ws_messages) > 10:
            st.session_state.ws_messages = st.session_state.ws_messages[:10]
        
        # Handle different message formats
        tick_data = None
        if isinstance(message, dict):
            # Format 1: {'tk': [{'ltp': ...}]}
            if 'tk' in message and isinstance(message.get('tk'), list) and len(message.get('tk', [])) > 0:
                tick_data = message['tk'][0]
                logger.debug(f"Found tick data in 'tk' format: {tick_data}")
            # Format 2: Direct tick data
            elif 'ltp' in message or 'lastTradedPrice' in message:
                tick_data = message
                logger.debug(f"Found direct tick data: {tick_data}")
            # Format 3: Nested in 'data'
            elif 'data' in message:
                tick_data = message['data']
                logger.debug(f"Found tick data in 'data': {tick_data}")
            # Format 4: Check if it's a list with tick data
            elif isinstance(message, list) and len(message) > 0:
                tick_data = message[0] if isinstance(message[0], dict) else None
                logger.debug(f"Found tick data in list format: {tick_data}")
        
        if tick_data:
            # Extract tick information with multiple field name variations
            ltp = float(tick_data.get('ltp', tick_data.get('lastTradedPrice', tick_data.get('LTP', 0))) or 0)
            ltq = float(tick_data.get('ltq', tick_data.get('lastTradedQty', tick_data.get('LTQ', 0))) or 0)
            volume = float(tick_data.get('v', tick_data.get('volume', tick_data.get('Volume', 0))) or 0)
            bid_price = float(tick_data.get('bp', tick_data.get('bidPrice', tick_data.get('BidPrice', 0))) or 0)
            bid_qty = float(tick_data.get('bq', tick_data.get('bidQty', tick_data.get('BidQty', 0))) or 0)
            ask_price = float(tick_data.get('ap', tick_data.get('askPrice', tick_data.get('AskPrice', 0))) or 0)
            ask_qty = float(tick_data.get('aq', tick_data.get('askQty', tick_data.get('AskQty', 0))) or 0)
            open_price = float(tick_data.get('o', tick_data.get('open', tick_data.get('Open', 0))) or 0)
            high = float(tick_data.get('h', tick_data.get('high', tick_data.get('High', 0))) or 0)
            low = float(tick_data.get('l', tick_data.get('low', tick_data.get('Low', 0))) or 0)
            close = float(tick_data.get('c', tick_data.get('close', tick_data.get('Close', ltp))) or ltp)
            
            tick_info = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                'ltp': ltp,
                'ltq': ltq,
                'volume': volume,
                'bid_price': bid_price,
                'bid_qty': bid_qty,
                'ask_price': ask_price,
                'ask_qty': ask_qty,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'change': close - open_price if open_price > 0 else 0
            }
            
            logger.info(f"Processed tick: LTP={ltp}, Volume={volume}")
            
            # Update tick count
            if 'tick_count' not in st.session_state:
                st.session_state.tick_count = 0
            st.session_state.tick_count += 1
            
            # Add to session state (keep last 100 ticks)
            if 'liveTicks' not in st.session_state:
                st.session_state.liveTicks = []
            
            # Insert at beginning and limit to 100
            st.session_state.liveTicks.insert(0, tick_info)
            if len(st.session_state.liveTicks) > 100:
                st.session_state.liveTicks = st.session_state.liveTicks[:100]
        else:
            logger.warning(f"Could not extract tick data from message: {message}")
    except Exception as e:
        logger.exception(f"Error processing tick: {e}")
        logger.debug(f"Message received: {message}")


def on_error(ws, error):
    """Callback function for WebSocket errors"""
    logger.error(f"WebSocket error: {error}")


def on_close(ws, code, reason):
    """Callback function for WebSocket close"""
    logger.info(f"WebSocket closed: {code} - {reason}")
    st.session_state.liveConnected = False


def on_open(ws):
    """Callback function for WebSocket open"""
    logger.info("WebSocket connection opened")
    # Subscribe after connection is established
    try:
        # Get subscription feed from the ws object if stored, or from session state
        feed = getattr(ws, '_subscription_feed', None)
        if not feed:
            try:
                feed = st.session_state.subscription_feed
            except:
                pass
        
        if feed:
            time.sleep(0.5)  # Small delay to ensure connection is ready
            ws.subscribe(feed)
            logger.info("Subscription sent after connection opened")
        else:
            logger.warning("No subscription feed found to subscribe")
    except Exception as e:
        logger.exception(f"Error subscribing after open: {e}")


def start_live_data_subscription(exchange, symbol_token):
    """Start WebSocket subscription for live tick data"""
    try:
        if not st.session_state.liveConnected:
            success, message = connect_to_live_api()
            if not success:
                return False, message
        
        # Close existing connection if any
        if st.session_state.ws:
            try:
                st.session_state.ws.close()
            except:
                pass
        
        # Create WebSocket connection
        token = st.session_state.liveAuthToken
        feed_token = st.session_state.liveFeedToken
        api_key = os.getenv('LIVE_DATA_API_KEY', '').strip()
        client_code = os.getenv('CLIENT_CODE', '').strip()
        
        # Convert symbol_token to string if needed
        symbol_token_str = str(symbol_token)
        
        # Prepare subscription feed
        feed = {
            "action": 1,  # Subscribe
            "mode": 1,    # LTP mode (1=LTP, 2=Full, 3=Quote)
            "tokenlist": [
                {
                    "exchangeType": exchange,
                    "tokens": [symbol_token_str]
                }
            ]
        }
        
        logger.info(f"Starting WebSocket for {exchange}:{symbol_token_str}")
        logger.info(f"Feed config: {feed}")
        
        ws = SmartWebSocketV2(
            auth_token=token,
            api_key=api_key,
            client_code=client_code,
            feed_token=feed_token
        )
        
        ws.on_open = on_open
        ws.on_message = on_ticks
        ws.on_error = on_error
        ws.on_close = on_close
        
        # Store feed in ws object for access in callbacks
        ws._subscription_feed = feed
        
        st.session_state.ws = ws
        st.session_state.subscription_feed = feed
        
        # Start WebSocket in a separate thread
        def run_ws():
            try:
                logger.info("Connecting WebSocket...")
                ws.connect()
                logger.info("WebSocket connected, subscription will be sent via on_open callback")
            except Exception as e:
                logger.exception(f"WebSocket thread error: {e}")
                st.session_state.ws_error = str(e)
        
        # Stop existing thread if running
        if st.session_state.wsThread and st.session_state.wsThread.is_alive():
            logger.info("WebSocket thread already running")
        else:
            ws_thread = threading.Thread(target=run_ws, daemon=True)
            ws_thread.start()
            st.session_state.wsThread = ws_thread
            logger.info("WebSocket thread started")
        
        return True, f"Live data subscription started for {exchange}:{symbol_token_str}"
    except Exception as e:
        logger.exception(f"Error starting live subscription: {e}")
        return False, str(e)


def get_historical_data(exchange, symboltoken, interval, fromdate, todate):
    """Fetch historical candle data"""
    try:
        if not st.session_state.connected:
            return None, "Not connected to API"
        
        smartApi = st.session_state.smartApi
        historicParam = {
            "exchange": exchange,
            "symboltoken": symboltoken,
            "interval": interval,
            "fromdate": fromdate,
            "todate": todate
        }
        
        response = smartApi.getCandleData(historicParam)
        
        if response['status'] == False:
            return None, f"API Error: {response}"
        
        return response.get('data', []), None
    except Exception as e:
        logger.exception(f"Error fetching data: {e}")
        return None, str(e)


def create_candlestick_chart(data, symbol_name):
    """Create candlestick chart from data"""
    if not data:
        return None
    
    # Convert data to DataFrame
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Create formatted timestamp strings for x-axis labels (to avoid gaps)
    # Format based on interval - show date+time for intraday, just date for daily
    df['timestamp_str'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=(f'{symbol_name} - Candlestick Chart', 'Volume')
    )
    
    # Candlestick chart - use timestamp strings to remove gaps (category mode)
    fig.add_trace(
        go.Candlestick(
            x=df['timestamp_str'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Volume chart
    colors = ['#ef5350' if df['close'].iloc[i] < df['open'].iloc[i] else '#26a69a' 
              for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df['timestamp_str'],
            y=df['volume'],
            name='Volume',
            marker_color=colors
        ),
        row=2, col=1
    )
    
    # Update layout - set x-axis type to category to remove gaps between candles
    fig.update_layout(
        title=f'{symbol_name} - Historical Data',
        xaxis_rangeslider_visible=False,
        height=800,
        showlegend=True,
        template='plotly_dark'
    )
    
    # Update x-axes to category type to remove gaps - candles will be evenly spaced
    # Limit number of ticks shown to avoid clutter
    num_ticks = min(20, len(df))  # Show max 20 labels
    tick_interval = max(1, len(df) // num_ticks)
    
    fig.update_xaxes(
        type='category',
        title_text="Time",
        tickangle=-45,
        tickmode='linear',
        tick0=0,
        dtick=tick_interval,
        row=2, col=1
    )
    fig.update_xaxes(
        type='category',
        tickangle=-45,
        tickmode='linear',
        tick0=0,
        dtick=tick_interval,
        row=1, col=1
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


# Main UI
st.title("📈 SmartAPI Candle Chart Viewer")
st.markdown("---")

# Sidebar for connection and settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Connection status
    if st.session_state.connected:
        st.success("✅ Connected to SmartAPI")
        if st.button("Disconnect"):
            st.session_state.connected = False
            st.session_state.smartApi = None
            st.rerun()
    else:
        st.warning("⚠️ Not Connected")
        if st.button("Connect to SmartAPI"):
            with st.spinner("Connecting..."):
                success, message = connect_to_api()
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
    
    st.markdown("---")
    
    # Exchange selection
    st.subheader("📊 Exchange Selection")
    exchange = st.selectbox(
        "Select Exchange",
        ["NSE", "BSE", "NFO", "MCX"],
        index=0
    )
    
    # Symbol input with dropdown list
    st.subheader("🔍 Symbol Details")
    
    # Indices dictionary (Token: Name)
    indices = {
        "99926000": "NIFTY 50 - Nifty 50 Index",
        "99926009": "NIFTY BANK - Bank Nifty",
        "99926001": "NIFTY IT - Nifty IT Index",
        "99926002": "NIFTY FMCG - Nifty FMCG Index",
        "99926003": "NIFTY PHARMA - Nifty Pharma Index",
        "99926004": "NIFTY AUTO - Nifty Auto Index",
        "99926005": "NIFTY METAL - Nifty Metal Index",
        "99926006": "NIFTY ENERGY - Nifty Energy Index",
        "99926007": "NIFTY INFRA - Nifty Infrastructure Index",
        "99926008": "NIFTY REALTY - Nifty Realty Index",
        "99926010": "NIFTY PSU BANK - Nifty PSU Bank Index",
        "99926011": "NIFTY PVT BANK - Nifty Private Bank Index",
        "99926012": "NIFTY FIN SERVICE - Nifty Financial Services",
        "99926013": "NIFTY MIDCAP 50 - Nifty Midcap 50",
        "99926014": "NIFTY MIDCAP 100 - Nifty Midcap 100",
        "99926015": "NIFTY SMALLCAP 100 - Nifty Smallcap 100",
        "99926016": "NIFTY NEXT 50 - Nifty Next 50",
        "99926017": "NIFTY 100 - Nifty 100 Index",
        "99926018": "NIFTY 200 - Nifty 200 Index",
        "99926019": "NIFTY 500 - Nifty 500 Index",
        "1": "SENSEX - BSE Sensex"
    }
    
    # Common stocks dictionary (Token: Name)
    common_symbols = {
        "3045": "SBIN - State Bank of India",
        "2885": "RELIANCE - Reliance Industries",
        "11536": "TCS - Tata Consultancy Services",
        "4085": "INFY - Infosys",
        "1594": "HDFCBANK - HDFC Bank",
        "1333": "ICICIBANK - ICICI Bank",
        "16675": "BHARTIARTL - Bharti Airtel",
        "4963": "WIPRO - Wipro",
        "317": "LT - Larsen & Toubro",
        "10604": "HINDUNILVR - Hindustan Unilever",
        "772": "ITC - ITC Limited",
        "910": "AXISBANK - Axis Bank",
        "5258": "SUNPHARMA - Sun Pharmaceutical",
        "881": "MARUTI - Maruti Suzuki",
        "11483": "TITAN - Titan Company",
        "1270": "KOTAKBANK - Kotak Mahindra Bank",
        "236": "ASIANPAINT - Asian Paints",
        "10940": "NESTLEIND - Nestle India",
        "467": "ULTRACEMCO - UltraTech Cement",
        "15083": "BAJFINANCE - Bajaj Finance"
    }
    
    # Combine all symbols
    all_symbols = {**indices, **common_symbols}
    
    # Create options list for selectbox with categories
    symbol_options = ["Custom"] + ["--- INDICES ---"] + list(indices.values()) + ["--- STOCKS ---"] + list(common_symbols.values())
    
    selected_symbol = st.selectbox(
        "Select Symbol",
        symbol_options,
        index=0,
        help="Select an index or stock from the list or choose Custom to enter manually"
    )
    
    # Initialize default values
    symbol_token = "3045"
    symbol_name = "SBIN"
    
    if selected_symbol == "Custom" or selected_symbol.startswith("---"):
        # Show custom input fields
        col1, col2 = st.columns(2)
        with col1:
            symbol_token = st.text_input(
                "Symbol Token",
                value="99926000" if selected_symbol.startswith("---") else "3045",
                help="Enter the symbol token"
            )
        with col2:
            symbol_name = st.text_input(
                "Symbol Name",
                value="NIFTY 50" if selected_symbol.startswith("---") else "SBIN",
                help="Enter symbol name for display"
            )
    else:
        # Extract token and name from selected symbol
        for token, name in all_symbols.items():
            if name == selected_symbol:
                symbol_token = token
                symbol_name = name.split(" - ")[0]  # Get just the symbol name
                break
        st.info(f"✅ Selected: **{symbol_name}** (Token: {symbol_token})")
    
    # Interval selection
    st.subheader("⏱️ Time Interval")
    interval = st.selectbox(
        "Select Interval",
        ["ONE_MINUTE", "FIVE_MINUTE", "FIFTEEN_MINUTE", "THIRTY_MINUTE", "ONE_HOUR", "ONE_DAY"],
        index=0
    )
    
    # Date range
    st.subheader("📅 Date Range")
    col1, col2 = st.columns(2)
    with col1:
        from_date = st.date_input(
            "From Date",
            value=datetime.now() - timedelta(days=7),
            max_value=datetime.now()
        )
        from_time = st.time_input("From Time", value=datetime.strptime("09:00", "%H:%M").time())
    with col2:
        to_date = st.date_input(
            "To Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
        to_time = st.time_input("To Time", value=datetime.strptime("16:00", "%H:%M").time())
    
    # Fetch button
    fetch_button = st.button("📥 Fetch Data", type="primary", use_container_width=True)

# Main content area
if not st.session_state.connected:
    st.info("👈 Please connect to SmartAPI using the sidebar to fetch data")
else:
    # Store selected symbol for live data
    current_symbol_key = f"{exchange}_{symbol_token}_{symbol_name}"
    
    # Start live data subscription if symbol changed
    if st.session_state.selectedSymbolForLive != current_symbol_key:
        st.session_state.selectedSymbolForLive = current_symbol_key
        st.session_state.liveTicks = []  # Clear old ticks
    
    # Live Tick Data Section
    if st.session_state.selectedSymbolForLive:
        st.subheader(f"🔴 Live Tick Data - {symbol_name} ({exchange})")
        
        # Status indicator
        if st.session_state.liveConnected:
            if st.session_state.wsThread and st.session_state.wsThread.is_alive():
                st.success("🟢 Live data connection active - WebSocket running")
            else:
                st.warning("🟡 Live API connected but WebSocket not running")
        else:
            st.info("⚪ Live data not connected")
        
        # Live Data Status Panel
        st.markdown("### 📊 Live Data Status")
        status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns(5)
        
        with status_col1:
            total_messages = st.session_state.get('message_count', 0)
            st.metric("Total Messages", f"{total_messages:,}")
        
        with status_col2:
            total_ticks = st.session_state.get('tick_count', 0)
            st.metric("Processed Ticks", f"{total_ticks:,}")
        
        with status_col3:
            stored_ticks = len(st.session_state.get('liveTicks', []))
            st.metric("Stored Ticks", f"{stored_ticks}")
        
        with status_col4:
            last_msg_time = st.session_state.get('last_message_time')
            if last_msg_time:
                time_str = last_msg_time.strftime("%H:%M:%S")
                st.metric("Last Message", time_str)
            else:
                st.metric("Last Message", "Never")
        
        with status_col5:
            last_msg_time = st.session_state.get('last_message_time')
            if last_msg_time:
                time_diff = (datetime.now() - last_msg_time).total_seconds()
                if time_diff < 5:
                    status_color = "🟢"
                    status_text = "Active"
                elif time_diff < 30:
                    status_color = "🟡"
                    status_text = "Slow"
                else:
                    status_color = "🔴"
                    status_text = "Stale"
                st.metric("Data Status", f"{status_color} {status_text}", f"{int(time_diff)}s ago")
            else:
                st.metric("Data Status", "⚪ No Data", "")
        
        # Show error if any
        if st.session_state.ws_error:
            st.error(f"WebSocket Error: {st.session_state.ws_error}")
            if st.button("Clear Error", key="clear_error"):
                st.session_state.ws_error = None
                st.rerun()
        
        # Start/Stop live data button
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            if st.button("▶️ Start Live Data", key="start_live"):
                # Reset counters when starting fresh
                st.session_state.message_count = 0
                st.session_state.tick_count = 0
                st.session_state.last_message_time = None
                success, message = start_live_data_subscription(exchange, symbol_token)
                if success:
                    st.success(message)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(message)
        
        with col2:
            if st.button("⏹️ Stop Live Data", key="stop_live"):
                if st.session_state.ws:
                    try:
                        st.session_state.ws.close()
                        st.session_state.liveConnected = False
                        st.success("Live data stopped")
                        time.sleep(0.5)
                        st.rerun()
                    except:
                        pass
        
        with col3:
            if st.button("🔄 Refresh / Clear Ticks", key="clear_ticks"):
                st.session_state.liveTicks = []
                st.rerun()
        
        # Debug section
        with st.expander("🔧 Debug Info"):
            debug_col1, debug_col2 = st.columns(2)
            with debug_col1:
                st.write("**Connection Status:**")
                st.write(f"- WebSocket Thread Alive: {st.session_state.wsThread.is_alive() if st.session_state.wsThread else False}")
                st.write(f"- Live Connected: {st.session_state.liveConnected}")
                st.write(f"- Total Messages Received: {st.session_state.get('message_count', 0):,}")
                st.write(f"- Total Ticks Processed: {st.session_state.get('tick_count', 0):,}")
                st.write(f"- Ticks in Memory: {len(st.session_state.get('liveTicks', []))}")
            
            with debug_col2:
                st.write("**Timing Info:**")
                last_msg = st.session_state.get('last_message_time')
                if last_msg:
                    time_diff = (datetime.now() - last_msg).total_seconds()
                    st.write(f"- Last Message: {last_msg.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
                    st.write(f"- Time Since Last: {time_diff:.2f} seconds")
                else:
                    st.write("- Last Message: Never received")
                    st.write("- Time Since Last: N/A")
            
            st.write("**Subscription Feed:**")
            st.json(st.session_state.subscription_feed if st.session_state.subscription_feed else {})
            
            if st.session_state.ws_messages:
                st.write("**Last WebSocket Messages (Raw):**")
                for msg in st.session_state.ws_messages[:5]:
                    st.text(f"{msg['timestamp']}: {msg['message']}")
            else:
                st.info("No WebSocket messages received yet")
            
            col_reset1, col_reset2 = st.columns(2)
            with col_reset1:
                if st.button("🔄 Reset Counters", key="reset_counters"):
                    st.session_state.message_count = 0
                    st.session_state.tick_count = 0
                    st.session_state.last_message_time = None
                    st.rerun()
            with col_reset2:
                if st.button("🗑️ Clear Debug Messages", key="clear_debug"):
                    st.session_state.ws_messages = []
                    st.rerun()
        
        # Display live tick data
        if st.session_state.liveTicks:
            # Latest tick info
            latest_tick = st.session_state.liveTicks[0] if st.session_state.liveTicks else None
            if latest_tick:
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("LTP", f"₹{latest_tick.get('ltp', 0):.2f}")
                with col2:
                    change = latest_tick.get('change', 0)
                    change_color = "normal" if change >= 0 else "inverse"
                    st.metric("Change", f"₹{change:.2f}", delta=f"{change:.2f}", delta_color=change_color)
                with col3:
                    st.metric("Volume", f"{latest_tick.get('volume', 0):,}")
                with col4:
                    st.metric("Bid", f"₹{latest_tick.get('bid_price', 0):.2f}")
                with col5:
                    st.metric("Ask", f"₹{latest_tick.get('ask_price', 0):.2f}")
            
            # Recent ticks table
            st.markdown("**Recent Ticks (Last 50):**")
            ticks_df = pd.DataFrame(st.session_state.liveTicks[:50])  # Show last 50 ticks
            if not ticks_df.empty:
                # Format the dataframe for better display
                display_df = ticks_df[['timestamp', 'ltp', 'ltq', 'volume', 'bid_price', 'ask_price', 'change']].copy()
                display_df.columns = ['Time', 'LTP', 'LTQ', 'Volume', 'Bid', 'Ask', 'Change']
                display_df['LTP'] = display_df['LTP'].apply(lambda x: f"₹{x:.2f}")
                display_df['Bid'] = display_df['Bid'].apply(lambda x: f"₹{x:.2f}")
                display_df['Ask'] = display_df['Ask'].apply(lambda x: f"₹{x:.2f}")
                display_df['Change'] = display_df['Change'].apply(lambda x: f"₹{x:.2f}")
                display_df['Volume'] = display_df['Volume'].apply(lambda x: f"{int(x):,}")
                
                st.dataframe(display_df, use_container_width=True, height=300)
        else:
            st.info("No live tick data yet. Click 'Start Live Data' to begin receiving ticks.")
        
        st.markdown("---")
    
    if fetch_button:
        # Combine date and time
        from_datetime = datetime.combine(from_date, from_time).strftime("%Y-%m-%d %H:%M")
        to_datetime = datetime.combine(to_date, to_time).strftime("%Y-%m-%d %H:%M")
        
        with st.spinner(f"Fetching data for {exchange}..."):
            data, error = get_historical_data(
                exchange=exchange,
                symboltoken=symbol_token,
                interval=interval,
                fromdate=from_datetime,
                todate=to_datetime
            )
            
            if error:
                st.error(f"Error: {error}")
            elif data:
                st.success(f"✅ Fetched {len(data)} candles")
                
                # Auto-start live data when historical data is fetched
                if st.session_state.selectedSymbolForLive:
                    success, message = start_live_data_subscription(exchange, symbol_token)
                    if success:
                        st.info("🟢 Live tick data subscription started automatically")
                
                # Display chart
                st.subheader(f"📊 Historical Chart - {symbol_name} ({exchange})")
                chart = create_candlestick_chart(data, f"{symbol_name} ({exchange})")
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                
                # Display data table
                with st.expander("📋 View Raw Data"):
                    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"{symbol_name}_{exchange}_{interval}_{from_date}_{to_date}.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("No data returned from API")

# Footer
st.markdown("---")
st.markdown("**SmartAPI Candle Chart Viewer** | Powered by SmartAPI-Python")


