Telegram AI Assistant Bot

This is a Telegram bot that leverages the NVIDIA API to provide an interactive chat experience with Large Language Models (LLMs), including multimodal capabilities (text and image understanding). The bot supports different chat modes, maintains conversation history, and can analyze code screenshots.
Features

    Multimodal Chat: Interact with the LLM using both text and images (e.g., send a screenshot of code for analysis).
    Chat Modes: Switch between "General Chat" and "Code Helper" modes, each with a predefined system prompt to tailor the LLM's behavior.
    Conversation History: The bot maintains a limited conversation history to provide context to the LLM. You can also view the chat history at any time.
    History Reset: Easily clear the current chat history for a fresh start in any mode.
    User-Friendly Interface: Utilizes Telegram's ReplyKeyboardMarkup for intuitive mode switching and feature access.
    Logging: Comprehensive logging for monitoring bot activity and debugging.

Getting Started

Follow these steps to set up and run your Telegram AI Assistant Bot.
Prerequisites

    Python 3.8+: Make sure you have a compatible Python version installed.
    Telegram Bot Token: Obtain a bot token from BotFather on Telegram.
    NVIDIA API Key: Get an API key from the NVIDIA AI Playground or your NVIDIA API dashboard.

Installation
1. Clone the repository:
  git clone https://github.com/rostik2008/LLama.git
  cd your-repo-name

2. Create a virtual environment (recommended):
  python -m venv venv
  source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
3. Install dependencies:
   pip install -r requirements.txt


   
Configuration
1. Set environment variables:
   The bot requires your Telegram Bot Token and NVIDIA API Key. Create a .env file in the root directory of your project or set these variables directly in your environment.
   TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
   NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"
   Replace YOUR_TELEGRAM_BOT_TOKEN and YOUR_NVIDIA_API_KEY with your actual tokens.
Running the Bot
Once configured, you can run the bot:
python bot.py



The bot will start polling for updates, and you should see a "Бот запущен. Ожидание сообщений..." (Bot launched. Waiting for messages...) message in your console.
Usage

Interact with the bot on Telegram:

    /start: Initializes the bot and displays the main keyboard.
    /reset: Clears the current conversation history for the active chat mode.
    /help: Displays a brief help message about the bot's functionalities.
    /history: Shows the recent conversation history for the current chat mode.
    Buttons:
        Общий чат (General Chat): Switches the bot to a general conversational mode.
        Помощник по коду (Code Helper): Switches the bot to a mode specialized in programming and code analysis.
        История чата (Chat History): Same as the /history command, displays the chat history.
    Send Text Messages: Simply type and send your questions or prompts.
    Send Photos: Attach an image (e.g., a code snippet screenshot). The bot will attempt to analyze it along with any caption you provide.
