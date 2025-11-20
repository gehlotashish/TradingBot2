"""
Diagnostic script for Telegram configuration
"""

import requests
import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '').strip()

print("=" * 60)
print("TELEGRAM CONFIGURATION DIAGNOSTIC")
print("=" * 60)
print()

print(f"Bot Token: {BOT_TOKEN[:30]}... (length: {len(BOT_TOKEN)})")
print(f"Chat ID: {CHAT_ID}")
print()

# Test 1: Verify bot token format
print("Test 1: Verifying bot token format...")
if ':' in BOT_TOKEN and len(BOT_TOKEN) > 30:
    print("[OK] Bot token format looks correct")
else:
    print("[WARNING] Bot token format may be incorrect")
    print("   Expected format: numbers:letters (e.g., 123456789:ABCdefGHIjklMNOpqrsTUVwxyz)")
print()

# Test 2: Get bot info
print("Test 2: Getting bot information...")
try:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    bot_info = response.json()
    if bot_info.get('ok'):
        bot_data = bot_info['result']
        print(f"[OK] Bot found successfully!")
        print(f"   Username: @{bot_data.get('username', 'N/A')}")
        print(f"   Name: {bot_data.get('first_name', 'N/A')}")
        print(f"   Bot ID: {bot_data.get('id', 'N/A')}")
    else:
        print(f"[ERROR] Bot API returned error: {bot_info}")
except Exception as e:
    print(f"[ERROR] Failed to get bot info: {e}")
    print("   The bot token is incorrect or invalid")
    exit(1)
print()

# Test 3: Test sending to chat
print("Test 3: Testing message sending...")
try:
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {
        'chat_id': CHAT_ID,
        'text': 'Test message from AI Trading Signal System',
    }
    response = requests.post(url, json=payload, timeout=10)
    response.raise_for_status()
    result = response.json()
    if result.get('ok'):
        print(f"[OK] Message sent successfully!")
        print(f"   Message ID: {result['result']['message_id']}")
        print(f"   Chat: {result['result']['chat'].get('title', result['result']['chat'].get('first_name', 'N/A'))}")
    else:
        error = result.get('description', 'Unknown error')
        error_code = result.get('error_code', 'N/A')
        print(f"[ERROR] Failed to send message")
        print(f"   Error Code: {error_code}")
        print(f"   Error: {error}")
        if error_code == 400:
            print("\n   Possible causes:")
            print("   - Chat ID is incorrect")
            print("   - Bot hasn't been started (send /start to the bot)")
        elif error_code == 403:
            print("\n   Possible causes:")
            print("   - Bot is blocked by the user")
            print("   - Bot hasn't been started (send /start to the bot)")
            print("   - Chat ID is for a group, but bot is not a member")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 403:
        print("[ERROR] 403 Forbidden - Bot cannot send messages")
        print("\n   SOLUTION:")
        print("   1. Open Telegram")
        print("   2. Search for your bot: @my_private2bot")
        print("   3. Click 'Start' or send /start command")
        print("   4. Run this test again")
    elif e.response.status_code == 400:
        print("[ERROR] 400 Bad Request")
        print("   - Chat ID might be incorrect")
        print("   - Try getting your chat ID from @userinfobot")
    else:
        print(f"[ERROR] HTTP Error: {e}")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")

print()
print("=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. Open Telegram on your phone/desktop")
print("2. Search for: @my_private2bot")
print("3. Click 'Start' button or send: /start")
print("4. Run this diagnostic again: python diagnose_telegram.py")
print("=" * 60)

