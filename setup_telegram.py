"""
Quick script to set up Telegram credentials in .env file
"""

import os

# Telegram credentials
TELEGRAM_BOT_TOKEN = "AAHJBLWXMDPagmVN2epGQOb5ig0xtdcb0_U"
TELEGRAM_CHAT_ID = "8055550422"

# Read existing .env if it exists
env_content = []
env_file = ".env"

if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        env_content = f.readlines()
    
    # Update Telegram credentials
    updated = False
    new_content = []
    for line in env_content:
        if line.startswith("TELEGRAM_BOT_TOKEN="):
            new_content.append(f"TELEGRAM_BOT_TOKEN={TELEGRAM_BOT_TOKEN}\n")
            updated = True
        elif line.startswith("TELEGRAM_CHAT_ID="):
            new_content.append(f"TELEGRAM_CHAT_ID={TELEGRAM_CHAT_ID}\n")
            updated = True
        else:
            new_content.append(line)
    
    # Add if not found
    if not updated:
        # Check if we need to add them
        has_token = any("TELEGRAM_BOT_TOKEN" in line for line in env_content)
        has_chat = any("TELEGRAM_CHAT_ID" in line for line in env_content)
        
        if not has_token or not has_chat:
            # Add at the end
            new_content.append(f"\n# Telegram Bot Configuration\n")
            if not has_token:
                new_content.append(f"TELEGRAM_BOT_TOKEN={TELEGRAM_BOT_TOKEN}\n")
            if not has_chat:
                new_content.append(f"TELEGRAM_CHAT_ID={TELEGRAM_CHAT_ID}\n")
    
    env_content = new_content
else:
    # Create new .env file
    env_content = [
        "# SmartAPI Configuration\n",
        "HISTORICAL_DATA_API_KEY=your_historical_api_key_here\n",
        "API_KEY=your_api_key_here\n",
        "LIVE_DATA_API_KEY=your_live_api_key_here\n",
        "\n",
        "# SmartAPI Credentials\n",
        "CLIENT_CODE=your_client_code\n",
        "PIN=your_pin\n",
        "TOTP_SECRET=your_totp_secret\n",
        "\n",
        "# Telegram Bot Configuration\n",
        f"TELEGRAM_BOT_TOKEN={TELEGRAM_BOT_TOKEN}\n",
        f"TELEGRAM_CHAT_ID={TELEGRAM_CHAT_ID}\n",
    ]

# Write .env file
try:
    with open(env_file, 'w') as f:
        f.writelines(env_content)
    print(f"[OK] Telegram credentials configured in {env_file}")
    print(f"   Bot Token: {TELEGRAM_BOT_TOKEN[:20]}...")
    print(f"   Chat ID: {TELEGRAM_CHAT_ID}")
    print("\n[NOTE] You still need to configure SmartAPI credentials in .env")
    print("   - HISTORICAL_DATA_API_KEY")
    print("   - LIVE_DATA_API_KEY")
    print("   - CLIENT_CODE")
    print("   - PIN")
    print("   - TOTP_SECRET")
except Exception as e:
    print(f"[ERROR] Error writing .env file: {e}")
    print("\nPlease create .env manually with:")
    print(f"TELEGRAM_BOT_TOKEN={TELEGRAM_BOT_TOKEN}")
    print(f"TELEGRAM_CHAT_ID={TELEGRAM_CHAT_ID}")

