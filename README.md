# Memory Telegram Bot

A clean and elegant Telegram bot that integrates with the Memory System API to provide intelligent memory storage and retrieval capabilities.

## Features

🧠 **Smart Memory Storage**: Save any message content (text, images, audio, video, documents) to your digital memory  
🔍 **Semantic Search**: Find relevant memories using natural language queries  
📋 **Item Retrieval**: Get specific memory items by their unique ID  
🤖 **Intelligent Processing**: Automatic content analysis and embedding generation  
⚡ **Fast & Reliable**: Built with modern async Python and robust error handling

## Commands

### `/remember`
Save content to your digital memory system.

**Usage:**
- Reply to any message with `/remember` to save it
- Use `/remember <text>` to save custom content
- Works with all media types (text, images, audio, video, documents)

**Examples:**
```
/remember Important meeting notes for tomorrow
```
```
[Reply to a message] /remember
```

### `/query <search terms>`
Search through your saved memories using AI-powered semantic search.

**Usage:**
```
/query project meeting notes
/query vacation photos from last summer
/query conversation about budget planning
```

### `/get_item <item_id>`
Retrieve a specific memory item by its unique ID.

**Usage:**
```
/get_item 550e8400-e29b-41d4-a716-446655440000
```

## Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) package manager
- Telegram Bot Token (from [@BotFather](https://t.me/botfather))
- Running Memory System API instance
- MinIO server for binary file storage

### Installation

1. **Clone and setup the project:**
```bash
git clone <your-repo>
cd tgbot
```

2. **Install dependencies:**
```bash
uv sync
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. **Set up your Telegram bot:**
   - Message [@BotFather](https://t.me/botfather) on Telegram
   - Create a new bot with `/newbot`
   - Copy the bot token to your `.env` file

### Configuration

Edit `.env` file with your settings:

```bash
# Required: Your bot token from BotFather
TELEGRAM_BOT_TOKEN=1234567890:AABBCCDDEEFFaabbccddeeff

# Memory API endpoint (default: localhost)
MEMORY_API_BASE_URL=http://localhost:8000

# MinIO Configuration for binary storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BUCKET_NAME=telegram-memories
MINIO_SECURE=false

# Optional: Bot settings
BOT_USERNAME=your_bot_username
ENABLE_LOGGING=true
LOG_LEVEL=INFO
```

### Running

```bash
uv run python main.py
```

The bot will automatically:
- Connect to your Memory API
- Set up command handlers
- Register bot commands in Telegram
- Start polling for messages

## Architecture

The bot follows clean architecture principles with clear separation of concerns:

```
src/
├── __init__.py          # Package initialization
├── models.py            # Pydantic data models
├── config.py            # Configuration management
├── memory_client.py     # HTTP client for Memory API
├── handlers.py          # Command handlers
└── bot.py              # Main bot application
```

### Key Design Principles

- **Immutable Data**: All models are immutable Pydantic classes
- **Async/Await**: Full async support for high performance
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Type Safety**: Full type hints throughout the codebase
- **Clean Code**: Functional programming style with clear separation of concerns

## Development

Run the bot locally:
```bash
uv run main.py
```

Test commands in Telegram:
1. Start chat with your bot
2. Try `/start` for welcome message
3. Use `/remember` to save content
4. Use `/query` to search
5. Use `/get_item` with returned IDs

## Memory API Integration

The bot integrates with the Memory System API providing:

- **Content Ingestion**: Saves messages with metadata
- **Semantic Search**: Finds relevant content using embeddings
- **Item Retrieval**: Gets specific items by ID
- **Background Processing**: Handles complex content types
- **Health Monitoring**: Checks API availability

### Supported Content Types

- **Text**: Plain text messages and captions
- **Images**: Photos with optional captions (stored in MinIO)
- **Audio**: Voice messages and audio files (stored in MinIO)
- **Video**: Video files with optional captions (stored in MinIO)
- **Documents**: File attachments (stored in MinIO)
- **Stickers**: Animated and static stickers (stored in MinIO)

All binary content is automatically downloaded from Telegram and stored in your MinIO instance for reliable access and backup.
