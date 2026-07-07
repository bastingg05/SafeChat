import logging
import requests
import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, MessageHandler, filters
import html

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# The token is now securely loaded from the .env file
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# The local URL of your SafeChat Flask server
SAFECHAT_API_URL = "http://127.0.0.1:5000/api/send"

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Intercepts messages, checks them with SafeChat, and deletes if offensive."""
    if not update.message or not update.message.text:
        return

    user = update.message.from_user
    username = user.first_name if user else "TelegramUser"
    text = update.message.text

    logging.info(f"Received message from {username}: {text}")

    # 1. Forward the message to the SafeChat AI Brain
    try:
        payload = {
            "text": text,
            "sender": "other",  # Treat as incoming
            "username": f"[Telegram] {username}",
            "is_audio": False
        }
        response = requests.post(SAFECHAT_API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # 2. Check if AI flagged it as offensive
            if result.get("is_offensive"):
                logging.warning(f"Offensive message detected! Label: {result.get('label')}")
                
                # Try to delete the original message (Bot must have delete permissions in the group)
                try:
                    await update.message.delete()
                    safe_username = html.escape(username)
                    # Create a permanent censor bar covering the entire length of the sentence
                    censored_text = "█" * len(text)
                    
                    spoiler_message = (
                        f"⚠️ <b>{safe_username}</b> sent an offensive message:\n"
                        f"{censored_text}"
                    )
                    
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=spoiler_message,
                        parse_mode="HTML"
                    )
                except Exception as e:
                    logging.error(f"Failed to delete/repost message. Does the bot have Admin rights? Error: {e}")
                    
    except Exception as e:
        logging.error(f"Failed to connect to SafeChat API: {e}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Intercepts voice notes, transcribes them, and checks for offensive content."""
    if not update.message or not update.message.voice:
        return
        
    user = update.message.from_user
    username = user.first_name if user else "TelegramUser"
    logging.info(f"Received voice note from {username}")

    try:
        # Download the voice file from Telegram
        file = await update.message.voice.get_file()
        audio_bytes = await file.download_as_bytearray()
        
        # Send to SafeChat API for Transcription
        transcribe_url = "http://127.0.0.1:5000/api/transcribe"
        files = {'audio': ('voice.ogg', bytes(audio_bytes), 'audio/ogg')}
        
        response = requests.post(transcribe_url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            transcribed_text = result.get("text", "").strip()
            
            if not transcribed_text:
                return
                
            logging.info(f"Transcribed voice note from {username}: {transcribed_text}")
            
            # Now send the transcribed text to the normal /api/send endpoint
            payload = {
                "text": transcribed_text,
                "sender": "other", 
                "username": f"[Telegram Audio] {username}",
                "is_audio": True
            }
            send_response = requests.post(SAFECHAT_API_URL, json=payload)
            
            if send_response.status_code == 200:
                send_result = send_response.json()
                
                # Check if AI flagged it as offensive
                if send_result.get("is_offensive"):
                    logging.warning(f"Offensive voice note detected! Label: {send_result.get('label')}")
                    
                    try:
                        await update.message.delete()
                        
                        safe_username = html.escape(username)
                        # Create a permanent censor bar covering the entire length of the voice transcript
                        censored_text = "█" * len(transcribed_text)
                        
                        spoiler_message = (
                            f"⚠️ <b>{safe_username}</b> sent an offensive voice note:\n"
                            f"{censored_text}"
                        )
                        
                        await context.bot.send_message(
                            chat_id=update.effective_chat.id,
                            text=spoiler_message,
                            parse_mode="HTML"
                        )
                    except Exception as e:
                        logging.error(f"Failed to delete/repost voice note. Error: {e}")
    except Exception as e:
        logging.error(f"Failed to process voice note: {e}")

if __name__ == '__main__':
    if TELEGRAM_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("ERROR: You must replace YOUR_TELEGRAM_BOT_TOKEN_HERE with your actual bot token!")
        exit(1)
        
    print("Starting SafeChat Telegram Bot...")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    
    # Listen for all text messages
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    
    # Listen for voice notes
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    
    print("Bot is listening! Press Ctrl+C to stop.")
    app.run_polling()
