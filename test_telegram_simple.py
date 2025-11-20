# Simple Telegram test
import requests
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '').strip()

print("Testing Telegram connection...")
print(f"Bot: @my_private2bot")
print(f"Chat ID: {CHAT_ID}")
print()

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
payload = {'chat_id': CHAT_ID, 'text': 'Test from AI Trading System'}

try:
    response = requests.post(url, json=payload, timeout=10)
    result = response.json()
    
    if result.get('ok'):
        print("SUCCESS! Message sent to Telegram.")
        print("Check your Telegram for the test message.")
    else:
        error_code = result.get('error_code', 'N/A')
        error_desc = result.get('description', 'Unknown')
        print(f"ERROR {error_code}: {error_desc}")
        
        if error_code == 403:
            print()
            print("SOLUTION:")
            print("1. Open Telegram")
            print("2. Search for: @my_private2bot")
            print("3. Click START or send /start")
            print("4. Run this test again")
except Exception as e:
    print(f"Error: {e}")

