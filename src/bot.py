import asyncio
from typing import Optional

from loguru import logger
from telegram import BotCommand, Update
from telegram.ext import Application, ApplicationBuilder, CommandHandler, ContextTypes

from src.components.cmd_handlers import BotCommandHandler
from src.components.memory_client import MemoryAPIClient
from src.components.minio_client import MinIOClient
from src.utils.config import BotConfig


class MemoryBot:
    def __init__(self, config: BotConfig) -> None:
        self._config = config
        self._memory_client: Optional[MemoryAPIClient] = None
        self._minio_client: Optional[MinIOClient] = None
        self._command_handler: Optional[BotCommandHandler] = None
        self._application: Optional[Application] = None

    async def initialize(self) -> None:
        self._memory_client = MemoryAPIClient(self._config.memory_api)

        self._minio_client = MinIOClient(self._config.minio)
        await self._minio_client.initialize()

        self._command_handler = BotCommandHandler(self._memory_client, self._minio_client)
        self._application = ApplicationBuilder().token(self._config.telegram_bot_token).build()

        if self._command_handler and self._application:
            self._command_handler.set_bot_instance(self._application.bot)

        await self._register_handlers()
        await self._set_bot_commands()

        logger.info("Memory bot initialized successfully")

    async def start(self) -> None:
        if not self._application:
            raise RuntimeError("Bot not initialized. Call initialize() first.")

        # Check API health
        if self._memory_client and not await self._memory_client.health_check():
            logger.warning("Memory API not working. Bot will start but may not function properly.")
        if self._minio_client and not self._minio_client.health_check():
            logger.warning("MinIO not working. Bot will start but may not store file properly.")

        if not self._application.updater:
            raise RuntimeError("Application has no updater configured.")

        logger.info("Starting Memory bot polling...")
        await self._application.initialize()
        # Start the background polling task
        await self._application.start()
        await self._application.updater.start_polling(allowed_updates=["message", "edited_message"])

    async def stop(self) -> None:
        if self._application:
            if self._application and self._application.updater:
                if self._application.updater.running:
                    await self._application.updater.stop()  # Stop the polling
            await self._application.stop()  # Stop the dispatcher
            await self._application.shutdown()  # Clean up application resources

        if self._memory_client:
            await self._memory_client.close()

        logger.info("Memory bot stopped")

    async def _register_handlers(self) -> None:
        if not self._application or not self._command_handler:
            return

        # Command handlers
        self._application.add_handler(
            CommandHandler("remember", self._command_handler.handle_remember_cmd)
        )
        self._application.add_handler(
            CommandHandler("query", self._command_handler.handle_query_cmd)
        )
        self._application.add_handler(
            CommandHandler("get_item", self._command_handler.handle_get_item_cmd)
        )

        # Help and start commands
        self._application.add_handler(CommandHandler("start", self._handle_start))
        self._application.add_handler(CommandHandler("help", self._handle_help))

    async def _set_bot_commands(self) -> None:
        if not self._application:
            return

        commands = [
            BotCommand("start", "Start the bot and get welcome message"),
            BotCommand("help", "Show help information"),
            BotCommand("remember", "Save a message or content to memory"),
            BotCommand("query", "Search through saved memories"),
            BotCommand("get_item", "Retrieve a specific memory item by ID"),
        ]

        await self._application.bot.set_my_commands(commands)

    async def _handle_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message:
            return

        welcome_text = (
            "🧠 **Welcome to Memory Bot!**\n\n"
            "I can help you save and search through your messages using advanced AI.\n\n"
            "**Available Commands:**\n"
            "• `/remember` - Save content to memory\n"
            "• `/query <search>` - Search your memories\n"
            "• `/get_item <id>` - Get specific memory item\n"
            "• `/help` - Show detailed help\n\n"
            "Ready to start building your digital memory!"
        )

        await update.message.reply_text(welcome_text, parse_mode="Markdown")

    async def _handle_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command."""
        if not update.message:
            return

        help_text = (
            "🔧 **Memory Bot Commands**\n\n"
            "**💾 /remember**\n"
            "Save content to your digital memory.\n"
            "• Reply to any message with `/remember` to save it\n"
            "• Use `/remember <text>` to save custom content\n"
            "• Works with text, images, audio, video, and documents\n\n"
            "**🔍 /query <search terms>**\n"
            "Search through your saved memories using AI.\n"
            "• Example: `/query project meeting notes`\n"
            "• Finds semantically similar content\n"
            "• Returns top 5 most relevant results\n\n"
            "**📋 /get_item <item_id>**\n"
            "Retrieve a specific memory item by its ID.\n"
            "• Use the ID from search results\n"
            "• Shows full item details and metadata\n\n"
            "**Tips:**\n"
            "• Memory items include metadata like timestamps and user info\n"
            "• Search works across all content types\n"
            "• All data is processed securely through the Memory API"
        )

        await update.message.reply_text(help_text, parse_mode="Markdown")


@logger.catch()
async def create_and_run_bot(config: Optional[BotConfig] = None) -> None:
    if config is None:
        config = BotConfig()  # type: ignore[call-arg]

    bot = MemoryBot(config)

    try:
        await bot.initialize()
        await bot.start()
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    except Exception as e:
        logger.error(f"Bot error: {e}")
        raise
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(create_and_run_bot())
