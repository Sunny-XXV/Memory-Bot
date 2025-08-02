import asyncio
import sys

from src.bot import create_and_run_bot


async def main() -> None:
    await create_and_run_bot()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Bot failed to start: {e}")
        sys.exit(1)
