import os
from dotenv import load_dotenv
from datetime import datetime
from zoneinfo import ZoneInfo

load_dotenv()

# Configuration variables
BOT_TOKEN = os.getenv("BOT_TOKEN") # Bot token
GUILD_ID = int(os.getenv("GUILD_ID")) # Server ID as integer
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") # Tavily API Key
HF_TOKEN = os.getenv("HF_TOKEN", "") # Optional


# Language for transcription: "en" for English, "pl" for Polish, etc.
LANGUAGE = "pl"

# Time zone for date and time formatting
ZONE = "Europe/Warsaw"

# Text-to-Speech voice
TTS_VOICE = "pl-PL-MarekNeural"

# If true, the models will run locally if possible
RUN_LOCALLY = False

# Enable or disable logging of transcriptions and responses
LOGGING = True 

# If false, the bot will transcribe all audio without requiring a trigger word.
REQUIRE_TRIGGER = True

# Silence threshold in seconds to consider as silence
SILENCE_THRESHOLD = 1.0

# Minimum audio length in seconds to consider for transcription
MIN_AUDIO_LENGTH = 0.6


# Trigger words
TRIGGERS = ["jarvis", "dlarwis", "jarewis", "elvis", "dziarowijs", "dziadowiz", "jarvan", "jarwis", "rarwis", "garmin", "jarvi", "garvis"] 

# Filtered phrases
COMMON_IGNORED_PHRASES = [
    "yhm", "mhm", "ahem", "khm", "khm khm", "ach", "eh",
    "mm", "mmh", "mmm", "hm", "hmm", "hmmm", "hmmmm", "hmmmmm" "uh", "uhm", 
    "uhmm", "uhmmm", "ehm", "ehmm", "ehmmm", "aha", "oho", 
    "ojej", "och", "oj", "hahaha", "hehehe", "hihihi", 
    "hohoho", ".", "?", "!",
]

zone = ZoneInfo(ZONE)
current_date = datetime.now(zone).strftime("%Y-%m-%d %H:%M:%S")

if LANGUAGE.lower() == "en":
    IGNORED_PHRASES = [
        "okey", "good", "alright", "fine", "yes", "no"
    ]
    INITIAL_PROMPT = "Jarvis, listen."
    SYSTEM_PROMPT = """
    You are called Jarvis/Garmin, you are an advanced AI assistant integrated into a Discord server. 
    Your primary function is toassist users by answering questions, providing information, and engaging
    in meaningful conversations. You should be polite, concise, and informative in your responses. 
    Always strive to understand the user's intent and provide accurate and relevant information.
    You have access to a tool "web_search" that allows you to search for information on the internet, which you can use 
    when you do not know the answer to a user's question or when you need the most up-to-date information.
    The most important: Respond briefly and concisely.
    """.strip()

elif LANGUAGE.lower() == "pl":
    IGNORED_PHRASES = [
        "okej", "dobra", "tak", "nie", "Wszelkie prawa zastrzeżone", "Dziękuję."
    ]
    INITIAL_PROMPT = "Jarvis, słuchaj."
    SYSTEM_PROMPT = f"""
    Nazywasz się Jarvis/Garmin, jesteś zaawansowanym asystentem AI zintegrowanym z serwerem Discord. 
    Twoją główną funkcją jest pomaganie użytkownikom poprzez odpowiadanie na pytania, dostarczanie informacji i 
    angażowanie się w znaczące rozmowy. Powinieneś być uprzejmy, zwięzły i informacyjny w swoich odpowiedziach. 
    Zawsze staraj się zrozumieć intencje użytkownika i dostarczać dokładne oraz istotne informacje.
    Masz dostęp do narzędzia "web_search" które pozwala ci wyszukiwać informacje w internecie, z którego możesz korzystać, 
    gdy nie znasz odpowiedzi na pytanie użytkownika lub gdy potrzebujesz najnowszych informacji.
    Najważniejsze: Odpowiadaj krótko i zwięźle. 
    Dzisiejsza data: {current_date}."
    """.strip()
else:
    IGNORED_PHRASES = []