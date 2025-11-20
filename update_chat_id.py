"""
Update Telegram Chat ID in .env file
"""

import os

# Your correct chat ID from Telegram
CORRECT_CHAT_ID = "1165952097"

env_file = ".env"

if not os.path.exists(env_file):
    print(f"[ERROR] .env file not found. Please create it first.")
    exit(1)

# Read existing .env
with open(env_file, 'r') as f:
    lines = f.readlines()

# Update chat ID
updated = False
new_lines = []
for line in lines:
    if line.startswith("TELEGRAM_CHAT_ID="):
        new_lines.append(f"TELEGRAM_CHAT_ID={CORRECT_CHAT_ID}\n")
        updated = True
    else:
        new_lines.append(line)

# If not found, add it
if not updated:
    new_lines.append(f"\n# Telegram Bot Configuration\n")
    new_lines.append(f"TELEGRAM_CHAT_ID={CORRECT_CHAT_ID}\n")

# Write back
with open(env_file, 'w') as f:
    f.writelines(new_lines)

print(f"[OK] Chat ID updated to: {CORRECT_CHAT_ID}")
print(f"   User: Ashish Gehlot (@Its_ashu_01)")
print()
print("Testing Telegram connection...")

# Test the connection
import requests
from dotenv import load_dotenv
load_dotenv()

BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '').strip()
CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '').strip()

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
payload = {
    'chat_id': CHAT_ID,
    'text': 'Test message from AI Trading Signal System - Configuration successful!'
}

try:
    response = requests.post(url, json=payload, timeout=10)
    result = response.json()
    
    if result.get('ok'):
        print("[SUCCESS] Test message sent successfully!")
        print("Check your Telegram for the test message.")
    else:
        error_code = result.get('error_code', 'N/A')
        error_desc = result.get('description', 'Unknown')
        print(f"[ERROR] {error_code}: {error_desc}")
except Exception as e:
    print(f"[ERROR] {e}")

