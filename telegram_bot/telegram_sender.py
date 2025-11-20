"""
Telegram Bot Sender Module
Sends trading signals to Telegram
"""

import requests
import os
import sys
from logzero import logger
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TELEGRAM_CONFIG, SIGNAL_CONFIG


class TelegramSender:
    """Sends messages to Telegram"""
    
    def __init__(self):
        self.bot_token = TELEGRAM_CONFIG['BOT_TOKEN']
        self.chat_id = TELEGRAM_CONFIG['CHAT_ID']
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self.last_signal_time = {}  # Track last signal time per symbol for cooldown
    
    def send_message(self, text, parse_mode='HTML'):
        """
        Send message to Telegram
        
        Args:
            text: Message text
            parse_mode: HTML or Markdown
        
        Returns:
            bool: True if successful
        """
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram bot token or chat ID not configured")
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Message sent to Telegram successfully")
            return True
            
        except Exception as e:
            logger.exception(f"Error sending message to Telegram: {e}")
            return False
    
    def format_signal_message(self, signal_data):
        """
        Format signal data into Telegram message
        
        Args:
            signal_data: Dictionary with signal information
        
        Returns:
            Formatted message string
        """
        symbol = signal_data.get('symbol', 'N/A')
        signal = signal_data.get('signal', 'N/A')
        confidence = signal_data.get('confidence', 0)
        current_price = signal_data.get('current_price', 0)
        trigger_price = signal_data.get('trigger_price', 0)
        reason = signal_data.get('reason', 'N/A')
        timestamp = signal_data.get('timestamp', datetime.now())
        
        # Format timestamp
        if isinstance(timestamp, datetime):
            time_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_str = str(timestamp)
        
        # Determine emoji based on signal
        signal_emoji = "🟢" if signal == "BUY" else "🔴"
        
        # Format confidence as percentage
        confidence_pct = confidence * 100
        
        # Build message
        message = f"""
<b>🤖 AI MODULE: ACTIVE</b>

📊 <b>Index:</b> {symbol}
💰 <b>Current Price:</b> ₹{current_price:.2f}
{signal_emoji} <b>Signal:</b> {signal}
🎯 <b>Trigger:</b> ₹{trigger_price:.2f}
📈 <b>Accuracy:</b> {confidence_pct:.1f}%

💡 <b>Reason:</b> {reason}

⏰ <b>Time:</b> {time_str}
"""
        
        return message.strip()
    
    def send_signal(self, signal_data, check_cooldown=True):
        """
        Send trading signal to Telegram
        
        Args:
            signal_data: Dictionary with signal information
            check_cooldown: If True, check cooldown period before sending
        
        Returns:
            bool: True if sent successfully
        """
        # Check confidence threshold
        confidence = signal_data.get('confidence', 0)
        if confidence < SIGNAL_CONFIG['MIN_CONFIDENCE']:
            logger.info(f"Signal confidence {confidence:.2%} below threshold {SIGNAL_CONFIG['MIN_CONFIDENCE']:.2%}")
            return False
        
        # Check cooldown
        if check_cooldown:
            symbol_key = signal_data.get('symbol_key', 'default')
            last_time = self.last_signal_time.get(symbol_key)
            
            if last_time:
                time_diff = (datetime.now() - last_time).total_seconds() / 60  # minutes
                if time_diff < SIGNAL_CONFIG['SIGNAL_COOLDOWN']:
                    logger.info(f"Signal cooldown active for {symbol_key}: {time_diff:.1f} minutes")
                    return False
        
        # Format and send message
        message = self.format_signal_message(signal_data)
        success = self.send_message(message)
        
        if success:
            # Update last signal time
            symbol_key = signal_data.get('symbol_key', 'default')
            self.last_signal_time[symbol_key] = datetime.now()
        
        return success
    
    def send_test_message(self):
        """Send test message to verify Telegram configuration"""
        test_message = """
<b>🤖 AI Trading Signal System</b>

✅ Telegram bot is configured correctly!

This is a test message to verify the connection.
"""
        return self.send_message(test_message)


if __name__ == "__main__":
    # Test Telegram sender
    sender = TelegramSender()
    
    # Send test message
    print("Sending test message...")
    if sender.send_test_message():
        print("✓ Test message sent successfully!")
    else:
        print("✗ Failed to send test message")
    
    # Test signal format
    test_signal = {
        'symbol': 'BANKNIFTY',
        'symbol_key': 'BANKNIFTY',
        'signal': 'BUY',
        'confidence': 0.75,
        'current_price': 45000.50,
        'trigger_price': 45100.00,
        'reason': 'RSI oversold, MACD bullish, EMA bullish crossover',
        'timestamp': datetime.now()
    }
    
    print("\n" + "="*50)
    print("SAMPLE SIGNAL MESSAGE:")
    print("="*50)
    print(sender.format_signal_message(test_signal))
    print("="*50)

