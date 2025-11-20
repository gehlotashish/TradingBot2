"""
Direct test of Telegram API to verify credentials
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

print(f"Bot Token: {BOT_TOKEN[:20]}...")
print(f"Chat ID: {CHAT_ID}")
print()

# Test 1: Get bot info
print("Test 1: Getting bot information...")
try:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    bot_info = response.json()
    if bot_info.get('ok'):
        print(f"[OK] Bot found: @{bot_info['result']['username']}")
        print(f"     Bot name: {bot_info['result']['first_name']}")
    else:
        print(f"[ERROR] Bot API error: {bot_info}")
except Exception as e:
    print(f"[ERROR] Failed to get bot info: {e}")
    print("   This usually means the bot token is incorrect")
    print("   Please verify the token from @BotFather")

print()

# Test 2: Send message
print("Test 2: Sending test message...")
try:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': 'Test message from AI Trading Signal System',
        'parse_mode': 'HTML'
    }
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    result = response.json()
    if result.get('ok'):
        print(f"[OK] Message sent successfully!")
        print(f"     Message ID: {result['result']['message_id']}")
    else:
        print(f"[ERROR] Failed to send message: {result}")
        if result.get('error_code') == 400:
            print("   This usually means:")
            print("   - Chat ID is incorrect")
            print("   - Bot hasn't been started (send /start to bot)")
            print("   - Bot was blocked by user")
except Exception as e:
    print(f"[ERROR] Failed to send message: {e}")

print()
print("=" * 60)
print("Troubleshooting:")
print("1. Verify bot token from @BotFather on Telegram")
print("2. Start the bot by sending /start command")
print("3. Verify chat ID (use @userinfobot to get your chat ID)")
print("4. Make sure bot is not blocked")
print("=" * 60)

