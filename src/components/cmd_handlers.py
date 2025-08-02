from typing import Any, Callable, List, Optional, Tuple
from uuid import UUID

from loguru import logger
from telegram import Message, Update
from telegram.ext import ContextTypes
from telegram.helpers import escape_markdown

from src.components.memory_client import MemoryAPIClient, MemoryAPIError
from src.components.minio_client import MinIOClient, MinIOClientError
from src.utils.models import MemoryItemRaw, RetrievalResponse


class BotCommandHandler:
    def __init__(self, memory_client: MemoryAPIClient, minio_client: MinIOClient) -> None:
        self._memory_client = memory_client
        self._minio_client = minio_client

    async def handle_remember_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message:
            return

        try:
            memory_item = await self._create_msg_memoryitem(update.message)
            if not memory_item:
                await update.message.reply_text(
                    "❌ No content to remember. Either reply to a message or include content in your message.",
                    parse_mode="Markdown",
                )
                return

            response = await self._memory_client.ingest(memory_item)

            await update.message.reply_text(
                f"✅ Memory saved:\n📝 Item ID: `{response.item_id}`\n🕒 Status: {response.status}"
            )

        except MemoryAPIError as e:
            logger.error(f"Memory API error in remember command: {e}")
            await update.message.reply_text(f"❌ Failed to save memory: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in remember command: {e}")
            await update.message.reply_text("❌ An unexpected error occurred while saving memory.")

    async def handle_query_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not context.args:
            if update.message:
                await update.message.reply_text(
                    "❓ Please provide a search query.\nUsage: `/query <search terms>`",
                    parse_mode="MarkdownV2",
                )
            return

        query = " ".join(context.args)

        try:
            response = await self._memory_client.retrieve(query, top_k=5)

            if not response.results:
                escaped_query = escape_markdown(query, version=2)
                await update.message.reply_text(
                    f"🔍 No memories found for: `{escaped_query}`", parse_mode="MarkdownV2"
                )
                return

            formatted_results = self._fmt_search_results(response, query)
            logger.debug(f"Formatted search results:\n{formatted_results}")
            await update.message.reply_text(formatted_results, parse_mode="MarkdownV2")

        except MemoryAPIError as e:
            logger.error(f"Memory API error in query command: {e}")
            await update.message.reply_text(f"❌ Search failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in query command: {e}", exc_info=True)
            await update.message.reply_text("❌ An unexpected error occurred during search.")

    async def handle_get_item_cmd(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.message or not context.args:
            if update.message:
                await update.message.reply_text(
                    "📋 Please provide an item ID.\nUsage: `/get_item <item_id>`",
                    parse_mode="Markdown",
                )
            return

        try:
            item_id = UUID(context.args[0])
            item = await self._memory_client.get_item(item_id)

            formatted_item = self._fmt_memory_item(item)
            await update.message.reply_text(formatted_item, parse_mode="Markdown")

        except ValueError:
            await update.message.reply_text(
                "❌ Invalid item ID format. Please provide a valid UUID."
            )
        except MemoryAPIError as e:
            logger.error(f"Memory API error in get_item command: {e}")
            await update.message.reply_text(f"❌ Failed to retrieve item: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in get_item command: {e}")
            await update.message.reply_text(
                "❌ An unexpected error occurred while retrieving the item."
            )

    async def _create_msg_memoryitem(self, message: Message) -> Optional[MemoryItemRaw]:
        mem_message = message.reply_to_message if message.reply_to_message else message

        content_type, data_uri = "text", None
        media_obj, media_type, file_name = None, None, None
        text_content = mem_message.text or mem_message.caption

        mapping: List[Tuple[str, str, Callable[[Any], str]]] = [
            ("photo", "image", lambda p: f"photo_{p.file_id}.jpg"),
            ("voice", "audio", lambda v: f"voice_{v.file_id}.ogg"),
            ("video", "video", lambda v: f"video_{v.file_id}.mp4"),
            ("document", "document", lambda d: d.file_name or f"document_{d.file_id}"),
            ("audio", "audio", lambda a: a.file_name or f"audio_{a.file_id}.mp3"),
            ("video_note", "video", lambda vn: f"video_note_{vn.file_id}.mp4"),
            ("sticker", "image", lambda s: f"sticker_{s.file_id}.webp"),
        ]
        for attr_name, media_type, get_filename in mapping:
            if raw_obj := getattr(mem_message, attr_name, None):
                content_type = media_type
                media_obj = raw_obj[-1] if attr_name == "photo" else raw_obj
                file_name = get_filename(media_obj)

        if content_type != "text":
            assert media_obj is not None, "No media content found in message"
            assert media_type is not None, "No media type specified"
            assert file_name is not None, "No file name generated for media content"

            try:
                data_uri = await self._store_telegram_file(media_obj.file_id, file_name)

            except (MinIOClientError, Exception) as e:
                logger.error(f"Failed to store binary content in MinIO: {e}")
                data_uri = f"telegram_{media_type}:{media_obj.file_id}"

        return MemoryItemRaw(
            content_type=content_type,
            text_content=text_content,
            data_uri=data_uri,
            event_timestamp=mem_message.date,
            meta=self._extract_message_meta(mem_message),
            reply_to_id=None,
        )

    async def _store_telegram_file(self, file_id: str, file_name: str) -> str:
        try:
            file = await self._get_telegram_bot().get_file(file_id)
            file_bytes = await file.download_as_bytearray()

            metadata = {
                "telegram_file_id": file_id,
                "original_file_path": file.file_path or "",
                "file_size": str(file.file_size) if file.file_size else "unknown",
            }

            return self._minio_client.store_file(
                data=bytes(file_bytes),
                file_path=file_name,
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Failed to store Telegram file {file_id}: {e}")
            raise

    def _extract_message_meta(self, message: Message) -> dict:
        metadata = {
            "source": "telegram",
            "message_id": message.message_id,
            "chat_id": message.chat.id,
            "chat_type": message.chat.type,
        }

        if message.from_user:
            metadata.update(
                {
                    "user_id": message.from_user.id,
                    "username": message.from_user.username,
                    "first_name": message.from_user.first_name,
                    "last_name": message.from_user.last_name,
                }
            )

        if message.chat.title:
            metadata["chat_title"] = message.chat.title

        return metadata

    def _fmt_search_results(self, response: RetrievalResponse, query: str) -> str:
        escaped_query = escape_markdown(query, version=2)
        result_text = f"🔍 *Search Results for:* `{escaped_query}`\n\n"

        for i, result in enumerate(response.results, 1):
            item = result.item
            score = result.score

            # Truncate content for display
            content = item.text_content or item.analyzed_text or "[No text content]"
            if len(content) > 300:
                content = content[:297] + "..."
            content = escape_markdown(content, version=2)
            content = "\n".join([f">{line}" for line in content.split("\n")])

            score = escape_markdown(f"{score:.2f}", version=2)
            timestamp = escape_markdown(item.event_timestamp.strftime("%Y-%m-%d %H:%M"), version=2)
            # content_type = escape_markdown(item.content_type, version=2)

            result_text += (
                f"*{i}\\. \\(Score: {score}\\)*\n{content}\n\n🕒 {timestamp}\n🆔 `{item.id}`\n\n"
                # f"📂 {content_type}\n\n"
            )

        return result_text

    def _fmt_memory_item(self, item) -> str:
        """Format a single memory item for display."""
        content = item.text_content or item.analyzed_text or "[No text content]"
        timestamp = item.event_timestamp.strftime("%Y-%m-%d %H:%M:%S")
        created = item.created_at.strftime("%Y-%m-%d %H:%M:%S")

        result = (
            f"📋 **Memory Item Details**\n\n"
            f"🆔 **ID:** `{item.id}`\n"
            # f"📂 **Type:** {item.content_type}\n"
            f"🕒 **Event Time:** {timestamp}\n"
            f"📅 **Created:** {created}\n\n"
            f"📝 **Content:**\n{content}\n"
        )

        if item.meta:
            result += f"\n🏷️ **Metadata:** `{item.meta}`"

        if item.data_uri:
            if self._is_minio_url(item.data_uri):
                result += "\n🔗 **Stored in MinIO:** Available for download"
            else:
                result += f"\n🔗 **Data URI:** `{item.data_uri}`"

        return result

    def _is_minio_url(self, url: str) -> bool:
        if not url:
            return False
        minio_config = self._minio_client._config
        protocol = "https" if minio_config.secure else "http"
        expected_prefix = f"{protocol}://{minio_config.endpoint}/{minio_config.bucket_name}/"
        return url.startswith(expected_prefix)

    def _get_telegram_bot(self):
        bot = getattr(self, "_bot_instance", None)
        if bot is None:
            raise RuntimeError("Bot instance not set. Call set_bot_instance() first.")
        return bot

    def set_bot_instance(self, bot):
        self._bot_instance = bot
