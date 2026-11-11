import os
import asyncio
import logging
import time
from dotenv import load_dotenv
from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from groq import AsyncGroq

load_dotenv()

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = AsyncGroq(api_key=GROQ_API_KEY)

# Groq Free Tier da faol va eng tezkor modellar:
FAST_MODEL = "openai/gpt-oss-20b"    # /oson - chaqmoqdek tez rejim
THINK_MODEL = "openai/gpt-oss-120b"  # /oyla - chuqur tahlil rejimi

SYSTEM_INSTRUCTION = """Sen juda aqlli, tezkor va aniq javob beradigan AI yordamchisan. 
Faqat o'zbek tilida, tushunarli, xatosiz va mantiqiy javob ber."""

histories: dict[int, list[dict]] = {}
user_modes: dict[int, bool] = {}  # False = /oson, True = /oyla
MAX_HISTORY = 20

def get_history(chat_id: int) -> list[dict]:
    return histories.setdefault(chat_id, [])

def trim_history(chat_id: int) -> None:
    h = histories.get(chat_id)
    if h and len(h) > MAX_HISTORY:
        histories[chat_id] = h[-MAX_HISTORY:]

async def keep_typing(chat_id: int, context: ContextTypes.DEFAULT_TYPE):
    try:
        while True:
            await context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    histories.pop(chat_id, None)
    user_modes[chat_id] = False
    await update.message.reply_text(
        "Salom! 👋 Men tezkor AI yordamchiman.\n\n"
        "⚡ /oson — Tezkor Javoblar Uchun Chuqur O'ylamaydi (hozir yoqilgan)\n"
        "🧠 /oyla — murakkab savollar uchun chuqur tahlil rejimi\n"
        "🔄 /reset — suhbat tarixini tozalash"
    )

async def set_easy_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_modes[update.effective_chat.id] = False
    await update.message.reply_text("⚡ **Oson (chaqmoqdek tez) rejim** yoqildi.", parse_mode="Markdown")

async def set_think_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_modes[update.effective_chat.id] = True
    await update.message.reply_text("🧠 **Chuqur o'ylash rejimi** yoqildi.", parse_mode="Markdown")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    histories.pop(update.effective_chat.id, None)
    await update.message.reply_text("✅ Suhbat tarixi tozalandi.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    user_text = update.message.text
    if not user_text:
        return

    is_thinking = user_modes.get(chat_id, False)
    selected_model = THINK_MODEL if is_thinking else FAST_MODEL
    effort = "high" if is_thinking else "low"

    history = get_history(chat_id)
    history.append({"role": "user", "content": user_text})
    trim_history(chat_id)

    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}] + histories[chat_id]

    typing_task = asyncio.create_task(keep_typing(chat_id, context))
    initial_text = "🧠 Chuqur tahlil qilyapman..." if is_thinking else "Javob Tayyorlanmoqda..."
    msg = await update.message.reply_text(initial_text)

    try:
        stream = await client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=0.6,
            max_tokens=3000,
            reasoning_effort=effort,
            stream=True,
        )

        final_text = ""
        last_edit_time = time.time()

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                final_text += delta
                if time.time() - last_edit_time > 1.0:
                    try:
                        await msg.edit_text(final_text + " Javob tayyorlanmoqda....")
                    except Exception:
                        pass
                    last_edit_time = time.time()

        if not final_text.strip():
            await msg.edit_text("(javob bo'sh)")
            return

        if len(final_text) > 4000:
            parts = [final_text[i:i+4000] for i in range(0, len(final_text), 4000)]
            await msg.edit_text(parts[0])
            for part in parts[1:]:
                await update.message.reply_text(part)
        else:
            await msg.edit_text(final_text)

        history.append({"role": "assistant", "content": final_text})
        trim_history(chat_id)

    except Exception as e:
        log.exception("Groq xatolik")
        await msg.edit_text(f"⚠️ Xatolik: {e}")
        history.pop()
    finally:
        typing_task.cancel()

def main() -> None:
    if not TELEGRAM_TOKEN or not GROQ_API_KEY:
        raise SystemExit("❌ .env da TELEGRAM_BOT_TOKEN yoki GROQ_API_KEY yo'q")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("oson", set_easy_mode))
    app.add_handler(CommandHandler("oyla", set_think_mode))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    log.info("Groq Bot ishga tushdi...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
