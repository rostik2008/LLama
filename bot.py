import logging
import base64
import io
import re
import os
import requests
import json
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import asyncio

# --- Настройки вашего бота и API ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

NVIDIA_MODEL_NAME = "meta/llama-4-maverick-17b-128e-instruct"
NVIDIA_INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# Настройка логирования для отслеживания ошибок и информации
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)

# --- Определения режимов чата и их системных промптов ---
MODE_GENERAL_CHAT = "Общий чат"
MODE_CODE_HELPER = "Помощник по коду"
BUTTON_HISTORY = "История чата"

SYSTEM_PROMPTS = {
    MODE_GENERAL_CHAT: "Вы полезный, дружелюбный и креативный ассистент, который может понимать текст и изображения.",
    MODE_CODE_HELPER: "Вы опытный программист и помощник по написанию кода, способный анализировать как текст, так и скриншоты кода (изображения). Предоставляйте решения, объяснения и примеры кода."
}

chat_states = {}
MAX_HISTORY_LENGTH = 10 

# --- Вспомогательные функции ---

def get_initial_history_for_mode(mode: str) -> list:
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS[MODE_GENERAL_CHAT])
    return [{"role": "system", "content": system_prompt}]

def create_main_keyboard() -> ReplyKeyboardMarkup:
    keyboard = [
        [KeyboardButton(MODE_GENERAL_CHAT), KeyboardButton(MODE_CODE_HELPER)],
        [KeyboardButton(BUTTON_HISTORY)]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=False)

async def process_llm_request(chat_id: int, user_message_content: str) -> str:
    if chat_id not in chat_states:
        chat_states[chat_id] = {
            "mode": MODE_GENERAL_CHAT,
            "history": get_initial_history_for_mode(MODE_GENERAL_CHAT)
        }

    chat_states[chat_id]["history"].append({"role": "user", "content": user_message_content})

    current_history = chat_states[chat_id]["history"]
    if len(current_history) > MAX_HISTORY_LENGTH + 1:
        chat_states[chat_id]["history"] = [current_history[0]] + \
                                          current_history[-(MAX_HISTORY_LENGTH):]

    messages_to_send = chat_states[chat_id]["history"]

    logging.info(f"Отправляем в LLM для chat_id {chat_id} (режим: {chat_states[chat_id]['mode']}): {messages_to_send}")

    headers = {
      "Authorization": f"Bearer {NVIDIA_API_KEY}",
      "Accept": "text/event-stream"
    }

    payload = {
      "model": NVIDIA_MODEL_NAME,
      "messages": messages_to_send,
      "max_tokens": 4096,
      "temperature": 0.6,
      "top_p": 0.95,
      "stream": True
    }

    full_response = []
    try:
        response = requests.post(NVIDIA_INVOKE_URL, headers=headers, json=payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode("utf-8")
                if decoded_line.startswith("data:"):
                    try:
                        json_data = json.loads(decoded_line[len("data:"):])
                        if "choices" in json_data and len(json_data["choices"]) > 0:
                            delta_content = json_data["choices"][0]["delta"].get("content")
                            if delta_content:
                                full_response.append(delta_content)
                    except json.JSONDecodeError as e:
                        logging.error(f"JSON Decode Error: {e} in line: {decoded_line}")
                        continue
        
        response_text = "".join(full_response)
        
        chat_states[chat_id]["history"].append({"role": "assistant", "content": response_text})
        logging.info(f"Получен ответ от LLM для chat_id {chat_id}: {response_text[:100]}...")
        
        return response_text

    except requests.exceptions.RequestException as e:
        logging.error(f"Ошибка HTTP запроса к LLM: {e}")
        return "Извините, произошла ошибка при получении ответа от модели (ошибка сети или API)."
    except Exception as e:
        logging.error(f"Неизвестная ошибка при обращении к LLM: {e}")
        return "Извините, произошла непредвиденная ошибка при получении ответа от модели."

# --- Обработчики команд Telegram-бота ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    chat_id = update.effective_chat.id

    chat_states[chat_id] = {
        "mode": MODE_GENERAL_CHAT,
        "history": get_initial_history_for_mode(MODE_GENERAL_CHAT)
    }

    await update.message.reply_html(
        f"Привет, {user.mention_html()}! Я бот, который может общаться с тобой. "
        f"Выбери режим чата с помощью кнопок, отправь мне сообщение или картинку!",
        reply_markup=create_main_keyboard()
    )
    logging.info(f"Команда /start получена от {user.id}")

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id
    if chat_id in chat_states:
        current_mode = chat_states[chat_id]["mode"]
        chat_states[chat_id]["history"] = get_initial_history_for_mode(current_mode)
        await update.message.reply_text(
            f"История разговора в режиме '{current_mode}' очищена. Мы можем начать новый диалог!",
            reply_markup=create_main_keyboard()
        )
        logging.info(f"История чата для {chat_id} (режим: {current_mode}) сброшена.")
    else:
        await update.message.reply_text(
            "История разговора пуста. Начните с /start.",
            reply_markup=create_main_keyboard()
        )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Я могу отвечать на твои сообщения и анализировать изображения. "
        "Используй команды /start, чтобы начать, /reset, чтобы очистить историю разговора. "
        "Кнопки внизу позволяют выбрать режим чата или просмотреть историю."
    )
    logging.info(f"Команда /help получена от {update.effective_user.id}")

async def set_mode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    chat_id = update.effective_chat.id

    if user_message in SYSTEM_PROMPTS:
        old_mode = chat_states[chat_id].get("mode", "Неизвестный")
        chat_states[chat_id]["mode"] = user_message
        chat_states[chat_id]["history"] = get_initial_history_for_mode(user_message)
        
        await update.message.reply_text(
            f"Режим чата изменен на **'{user_message}'**. История разговора сброшена. "
            f"Теперь я настроен как:\n__{SYSTEM_PROMPTS[user_message]}__",
            parse_mode="Markdown",
            reply_markup=create_main_keyboard()
        )
        logging.info(f"Пользователь {chat_id} изменил режим с '{old_mode}' на '{user_message}'.")

async def show_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = update.effective_chat.id

    if chat_id not in chat_states or len(chat_states[chat_id]["history"]) <= 1:
        await update.message.reply_text(
            "История чата пуста. Начните диалог!",
            reply_markup=create_main_keyboard()
        )
        return

    history_messages_display = []
    for msg in chat_states[chat_id]["history"][1:]: 
        role_display = "Вы" if msg["role"] == "user" else "Бот"
        content_for_display = msg["content"]

        if msg["role"] == "user" and '<img src="data:image' in content_for_display:
            content_for_display = re.sub(r'<img src="data:image[^>]+">', '[Изображение]', content_for_display)
            if not content_for_display.strip():
                content_for_display = '[Изображение]'
        
        history_messages_display.append(f"**{role_display}**: {content_for_display}")
    
    formatted_history = "\n\n".join(history_messages_display)
    
    await update.message.reply_text(
        f"**Ваша история чата (режим: {chat_states[chat_id]['mode']}):**\n\n{formatted_history}",
        parse_mode="Markdown",
        reply_markup=create_main_keyboard()
    )
    logging.info(f"История чата показана для {chat_id}.")

async def echo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_message = update.message.text
    chat_id = update.effective_chat.id
    logging.info(f"Получено текстовое сообщение от {chat_id}: {user_message}")

    if not user_message:
        await update.message.reply_text("Пустое сообщение.")
        return

    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    llm_response = await process_llm_request(chat_id, user_message)

    await update.message.reply_text(llm_response)
    logging.info(f"Отправлен текстовый ответ пользователю {chat_id}.")

async def handle_photo_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик для фото сообщений."""
    chat_id = update.effective_chat.id
    logging.info(f"Получено фото сообщение от {chat_id}.")

    await context.bot.send_chat_action(chat_id=chat_id, action="upload_photo")
    
    selected_base64_image = None
    
    try:
        # Telegram предоставляет несколько размеров фото, от самого маленького до самого большого.
        # Мы перебираем их, чтобы найти самый большой размер, который не превышает лимит.
        for photo_size in update.message.photo:
            photo_file_obj = await photo_size.get_file()
            photo_bytes_io = io.BytesIO()
            await photo_file_obj.download_to_memory(photo_bytes_io)
            photo_bytes_io.seek(0) # Перемещаем указатель в начало потока
            
            current_base64_image = base64.b64encode(photo_bytes_io.read()).decode('utf-8')
            
            # Лимит NVIDIA API для встроенных изображений: 180,000 символов Base64
            if len(current_base64_image) < 180_000:
                selected_base64_image = current_base64_image
            else:
                # Если текущий размер уже слишком большой, то и все последующие (большие)
                # также будут слишком большими, поэтому можно прекратить проверку.
                break 
        
        if selected_base64_image is None:
            logging.warning(f"Все изображения для {chat_id} слишком велики для обработки.")
            await update.message.reply_text(
                "Извините, все доступные размеры изображения слишком велики для обработки. "
                "Пожалуйста, попробуйте отправить изображение меньшего размера или с худшим качеством."
            )
            return

        # Формируем контент для LLM в виде одной строки
        image_html_tag = f'<img src="data:image/jpeg;base64,{selected_base64_image}" />'
        
        user_message_content = ""
        if update.message.caption:
            user_message_content = update.message.caption.strip() + " "
        
        user_message_content += image_html_tag

        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        llm_response = await process_llm_request(chat_id, user_message_content)
        await update.message.reply_text(llm_response)
        logging.info(f"Отправлен мультимодальный ответ пользователю {chat_id}.")

    except Exception as e:
        logging.error(f"Ошибка при обработке фото сообщения: {e}")
        await update.message.reply_text(
            "Извините, произошла ошибка при обработке изображения. Попробуйте еще раз или отправьте текст.",
            reply_markup=create_main_keyboard()
        )

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logging.error("Exception while handling an update:", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        await update.effective_message.reply_text(
            "Произошла непредвиденная ошибка. Пожалуйста, попробуйте снова или используйте /reset.",
            reply_markup=create_main_keyboard()
        )

def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("history", show_history_command))

    application.add_handler(MessageHandler(filters.TEXT & filters.Regex(f'^{MODE_GENERAL_CHAT}$|^{MODE_CODE_HELPER}$'), set_mode_handler))
    application.add_handler(MessageHandler(filters.TEXT & filters.Regex(f'^{BUTTON_HISTORY}$'), show_history_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo_message))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, echo_message))

    application.add_error_handler(error_handler)

    logging.info("Бот запущен. Ожидание сообщений...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()