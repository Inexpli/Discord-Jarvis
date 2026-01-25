import os
from dotenv import load_dotenv

load_dotenv()

# Configuration variables
BOT_TOKEN = os.getenv("BOT_TOKEN") # Bot token
GUILD_ID = int(os.getenv("GUILD_ID")) # Server ID as integer
HF_TOKEN = os.getenv("HF_TOKEN", "") # Optional


# Language for transcription: "en" for English, "pl" for Polish, etc.
LANGUAGE = "en"

# Enable or disable logging of transcriptions
LOGGING = True 

# If false, the bot will transcribe all audio without requiring a trigger word.
REQUIRE_TRIGGER = True

# Silence threshold in seconds to consider as silence
SILENCE_THRESHOLD = 1.0

# Minimum audio length in seconds to consider for transcription
MIN_AUDIO_LENGTH = 0.6


# Trigger words
TRIGGERS = ["jarvis", "dlarwis", "jarewis", "elvis", "dziarowijs", "dziadowiz", "jarvan", "jarwis", "rarwis", "garmin", "jarvi"] 

# Filtered phrases
COMMON_IGNORED_PHRASES = [
    "yhm", "mhm", "ahem", "khm", "khm khm", "ach", "eh",
    "mm", "mmh", "mmm", "hm", "hmm", "hmmm", "hmmmm", "hmmmmm" "uh", "uhm", 
    "uhmm", "uhmmm", "ehm", "ehmm", "ehmmm", "aha", "oho", 
    "ojej", "och", "oj", "hahaha", "hehehe", "hihihi", 
    "hohoho", ".", "?", "!"
]

if LANGUAGE.lower() == "en":
    IGNORED_PHRASES = [
        "okey", "good", "alright", "fine", "yes", "no"
    ]
    INITIAL_PROMPT = "Jarvis, listen."
elif LANGUAGE.lower() == "pl":
    IGNORED_PHRASES = [
        "okej", "dobra", "tak", "nie"
    ]
    INITIAL_PROMPT = "Jarvis, słuchaj."
else:
    IGNORED_PHRASES = []