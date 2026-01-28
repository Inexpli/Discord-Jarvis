# 🤖 AI Voice Assistant Jarvis for Discord

An advanced, real-time voice assistant integrated into Discord. This bot combines **Speech-to-Text (STT)**, **Large Language Models (LLM)**, and **Text-to-Speech (TTS)** to engage in natural, meaningful conversations with users in voice channels.

It features tool usage (Web Search, Time), silence detection, and dual-language support (English/Polish).

## ✨ Key Features

* **⚡ Real-time Transcription (STT):**
    * Supports **Local** inference using `Faster-Whisper` (CUDA recommended).
    * Supports **Cloud** inference using Groq API (Whisper V3) for ultra-low latency.
* **🧠 Intelligent Responses (LLM):**
    * Powered by **Llama 3 70B** via Groq API.
    * Context-aware conversations with memory management.
* **🗣️ Natural Voice (TTS):**
    * High-quality voice synthesis using `edge-tts` (Microsoft Azure Neural voices).
* **🛠️ Autonomous Tools:**
    * **Web Search:** Fetches real-time data (news, weather, facts) using Tavily API.
    * **Time Check:** Provides accurate local time and date.
* **🎙️ Smart Audio Handling:**
    * **VAD (Voice Activity Detection):** Automatically detects silence to process speech.
    * **Wake Words:** Configurable trigger words (e.g., "Jarvis", "Garmin") to activate the bot (optional).

## 🛠️ Prerequisites

* **Python 3.10+**
* **FFmpeg**: Essential for audio processing. Must be installed and added to your system's PATH.
    * *Windows:* `winget install ffmpeg`
    * *Linux:* `sudo apt install ffmpeg`
* **API Keys**: You will need keys for Discord, Groq, and Tavily.

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Inexpli/Discord-Jarvis
    cd Discord-Jarvis
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configuration (.env):**
    Create a `.env` file in the root directory and add your credentials:
    ```env
    BOT_TOKEN=your_discord_bot_token
    GUILD_ID=your_server_id
    GROQ_API_KEY=your_groq_api_key
    TAVILY_API_KEY=your_tavily_api_key
    HF_TOKEN=optional_huggingface_token
    ```

4.  **Customize Settings:**
    Open `config.py` to adjust:
    * `LANGUAGE`: Set to `"en"` for English or `"pl"` for Polish.
    * `RUN_LOCALLY`: Set to `True` to use local GPU resources, or `False` to use Groq API.
    * `TRIGGERS`: Add or remove wake words.

## 🚀 Usage

1.  **Start the bot:**
    ```bash
    python main.py
    ```
2.  **Discord Commands:**
    * **`/join`**: The bot joins your current voice channel and starts listening.
    * **`/stop`**: The bot leaves the channel.
3.  **Interaction:**
    * If `REQUIRE_TRIGGER = True`, start your sentence with "Jarvis" (or other configured triggers).
    * If `REQUIRE_TRIGGER = False`, the bot will respond to all speech detected.
    * The bot will listen, process your request, and reply via voice.

## ⚙️ Project Structure

* `main.py`: Core logic, Discord event handling, audio processing pipeline, and LLM integration.
* `config.py`: Configuration parameters, prompt templates, and environment variable loading.
* `.env`: storage for sensitive API keys (excluded from version control).

## 📋 Requirements (requirements.txt)

Ensure your `requirements.txt` includes the following libraries:

```text
py-cord
faster-whisper
python-dotenv
groq
tavily-python
edge-tts
numpy
```

## 📄 License
This project is licensed under the MIT License.

## ⚠️ Limitations
**Single Instance:** The bot currently utilizes global variables for conversation state. It is designed to work on one server/channel at a time. Running it on multiple servers simultaneously may cause conversation history overlap.

**Local Performance:** If RUN_LOCALLY = True, a decent GPU (NVIDIA) is required for Faster-Whisper to run smoothly.