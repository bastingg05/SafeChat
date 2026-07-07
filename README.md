# SafeChat

SafeChat is a powerful, context-aware hate speech detection platform specifically designed for Malayalam (Code-Mixed) language. It features a real-time web interface, a Telegram bot integration, and an AI brain powered by MuRIL (Multilingual Representations for Indian Languages).

## Features
- **Context-Aware AI**: Powered by a fine-tuned MuRIL model trained on a massive dataset to deeply understand Malayalam slang and context.
- **Telegram Bot Integration**: Automatically monitors Telegram groups and intercepts toxic text and voice notes.
- **Voice Note Processing**: Uses Whisper STT (Speech-to-Text) to accurately transcribe and moderate voice messages.
- **Real-time Keyword Gating**: Instant detection of extreme profanity, dynamically mapped across transliterations and native scripts (e.g., "വാണം").
- **Continuous Reinforcement Learning**: The AI learns from user feedback in the Web UI and automatically retrains itself on shutdown to constantly improve its accuracy.
- **Censor Bars**: Securely censors complete toxic sentences natively within Telegram to prevent tap-to-reveal workarounds.

## Architecture
- `chat_server.py`: Flask-based API server handling real-time chat, AI predictions, Whisper transcription, and feedback loop management.
- `telegram_bot.py`: The live Telegram bot that intercepts messages, communicates with the SafeChat API, and enforces community guidelines.
- `preprocessing.py`: Deep text standardization mapping casual conversational Malayalam (e.g., "poda", "da") into neutral vocabulary to prevent AI bias.
- `train_model.py` / `update_brain.py`: Scripts handling the automated dataset holdout splitting and Full Parameter Fine-Tuning of the MuRIL brain.

## Setup Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Environment Variables**
   Create a `.env` file in the root directory and add your Telegram bot token:
   ```env
   TELEGRAM_TOKEN=your_token_here
   ```

3. **Start the SafeChat Brain & Web Interface**
   ```bash
   python chat_server.py
   ```

4. **Start the Telegram Bot**
   ```bash
   python telegram_bot.py
   ```
