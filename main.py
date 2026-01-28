import discord
from discord.ext import tasks
from faster_whisper import WhisperModel
import io
import wave
import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from config import *
from groq import Groq, BadRequestError
from tavily import TavilyClient
import edge_tts
import json
from datetime import datetime
from zoneinfo import ZoneInfo

print("Starting bot...")

bot = discord.Bot(debug_guilds=[GUILD_ID])

thread_pool = ThreadPoolExecutor(max_workers=3)
model_lock = asyncio.Lock()
bot_controllers = {}

groq = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

IS_SPEAKING = False # Don't change this - used to TTS state management

conversation_history = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Get current information from the internet (weather, news, facts). Use this when the user asks about something that requires up-to-date knowledge.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to send to the search engine (e.g. 'current weather in Warsaw', 'who won the match yesterday')."
                    }
                },
                "required": ["query"],
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Fetch the current date and time in the user's local time zone.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }
    }
]



async def finished_callback(sink, channel: discord.TextChannel, *args):
    """
    Empty callback required by discord.py but not used.

    Args:
        sink: The audio sink.
        channel: The text channel to send messages to.
        *args: Additional arguments.
    """
    pass


class AutoCutSink(discord.sinks.PCMSink):
    def __init__(self, dest_channel):
        super().__init__()
        self.dest_channel = dest_channel
        self.user_data_buffer = {}     
        self.last_spoken_time = {}   
        self.packets_received = 0
        self.lock = threading.Lock()

    def write(self, data, user):
        
        if IS_SPEAKING: 
            return

        try:
            if not data: 
                return
            
            if user is None: 
                return

            user_id = user.id if hasattr(user, 'id') else int(user)

            with self.lock:
                if user_id not in self.user_data_buffer:
                    self.user_data_buffer[user_id] = bytearray()
                
                if hasattr(data, 'data'):
                    self.user_data_buffer[user_id].extend(data.data)
                else:
                    self.user_data_buffer[user_id].extend(data)

                self.last_spoken_time[user_id] = time.time()
                self.packets_received += 1

        except Exception as e:
            print(f"Write Error: {e}")


@tasks.loop(seconds=0.5)
async def check_silence_task():
    """
    Checks if users have stopped speaking and triggers transcription.
    """
    try:
        if IS_SPEAKING:
            return

        for guild in bot.guilds:
            vc = guild.voice_client
            if not vc or not vc.recording or not isinstance(vc.sink, AutoCutSink):
                continue
            
            sink = vc.sink
            current_time = time.time()

            target_text_channel = sink.dest_channel
            
            with sink.lock:
                active_users = list(sink.last_spoken_time.keys())
                to_process = []

                for user_id in active_users:
                    last_seen = sink.last_spoken_time[user_id]
                    if current_time - last_seen > SILENCE_THRESHOLD:
                        audio_data = sink.user_data_buffer.pop(user_id, None)
                        del sink.last_spoken_time[user_id]
                        
                        if audio_data and len(audio_data) >= 192000 * MIN_AUDIO_LENGTH:
                            to_process.append((user_id, audio_data))

            for user_id, audio_data in to_process:
                asyncio.create_task(process_transcription(guild, user_id, audio_data, target_text_channel))

    except Exception as e:
        print(f"Silence Loop Error: {e}")


async def speak_response(vc, text):
    """
    Converts text to speech and plays it in the voice channel.
    
    Args:
        vc: The voice client.
        text: The text to convert to speech.
    """

    global IS_SPEAKING
    if not vc or not vc.is_connected(): 
        return
    
    IS_SPEAKING = True
    
    communicate = edge_tts.Communicate(text, TTS_VOICE)
    filename = "response.mp3"
    await communicate.save(filename)

    if vc.is_playing():
        vc.stop()
    
    await asyncio.sleep(0.1)

    def after_tts(error):
        global IS_SPEAKING
        if error: 
            print(f"TTS Error: {error}")
        
        IS_SPEAKING = False
        play_keep_alive(vc)

    try:
        source = discord.FFmpegPCMAudio(filename)
        vc.play(source, after=after_tts)
    except Exception as e:
        print(f"Play Error: {e}")
        IS_SPEAKING = False
        play_keep_alive(vc)


async def process_transcription(guild, user_id, raw_pcm, channel):
    """
    Processes raw audio data into text and sends it to Discord.

    Args:
        guild: The Discord guild (server).
        user_id: The ID of the user who spoke.
        raw_pcm: The raw PCM audio data.
        channel: The Discord text channel to send the transcription to.
    """

    try:
        out_buffer = io.BytesIO()
        with wave.open(out_buffer, 'wb') as wav_file:
            wav_file.setnchannels(2)
            wav_file.setsampwidth(2)
            wav_file.setframerate(48000)
            wav_file.writeframes(raw_pcm)
        out_buffer.seek(0)

        username = f"User {user_id}"
        member = guild.get_member(user_id)
        if member: username = member.display_name

        if RUN_LOCALLY:
            model = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
            async with model_lock:
                def run_whisper():
                    segments, info = model.transcribe(
                        out_buffer, 
                        beam_size=5, 
                        language=LANGUAGE.lower(), 
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        initial_prompt=INITIAL_PROMPT if REQUIRE_TRIGGER else None 
                    )
                    return "".join([segment.text for segment in segments]).strip()
        else:
            def run_whisper():
                transcription = groq.audio.transcriptions.create(
                    file=("audio.wav", out_buffer.read()), 
                    model="whisper-large-v3-turbo",
                    prompt=INITIAL_PROMPT if REQUIRE_TRIGGER else None,
                    temperature=0.0, 
                    language=LANGUAGE.lower(), 
                    response_format="json"
                )
                return transcription.text.strip()
            
        text = await bot.loop.run_in_executor(thread_pool, run_whisper)

        if text:
            clean_text = text.lower().replace(",", "").replace(".", "").replace("?", "").strip()
            if clean_text in IGNORED_PHRASES or len(clean_text) < 6: 
                return

            if REQUIRE_TRIGGER:
                is_triggered = any(trigger in clean_text for trigger in TRIGGERS)
                if not is_triggered:
                    print(f"Ignoring: '{text}'")
                    return 
            
            if channel:
                if channel.permissions_for(guild.me).send_messages:
                    if LOGGING:
                        embed = discord.Embed(description=f"{text}", color=discord.Color.green())
                        embed.set_author(name=username, icon_url=member.avatar.url if member else None)
                        await channel.send(embed=embed)
                    
                    conversation_history.append({"role": "user", "content": text})

                    def ask_groq():
                        try:
                            response = groq.chat.completions.create(
                                messages=conversation_history,
                                model="llama-3.3-70b-versatile",
                                temperature=0.7, 
                                max_completion_tokens=300, 
                                stream=False,
                                tools=tools_schema, 
                                tool_choice="auto"
                            )
                        except BadRequestError as e:
                            print(f"BadRequestError: {e}")
                            response = groq.chat.completions.create(
                                messages=conversation_history,
                                model="llama-3.3-70b-versatile",
                                temperature=0.7,
                                max_completion_tokens=300,
                                stream=False
                            )
                        response_message = response.choices[0].message
                        tool_calls = response_message.tool_calls

                        if tool_calls:
                            conversation_history.append(response_message)
                            for tool_call in tool_calls:
                                if tool_call.function.name == "web_search":
                                    try:
                                        function_args = json.loads(tool_call.function.arguments)
                                        query = function_args.get("query")
                                        print(f"Searching: {query}")
                                        raw_result = tavily_client.search(query, search_depth="basic", max_tokens=500)
                                        search_result = json.dumps(raw_result, ensure_ascii=False)
                                    except Exception as e:
                                        search_result = f"Error: {e}"
                                    
                                    conversation_history.append({
                                        "tool_call_id": tool_call.id, 
                                        "role": "tool",
                                        "name": "web_search",
                                        "content": search_result,
                                    })
                                if tool_call.function.name == "current_time":
                                    try:
                                        print(f"Fetching current server time")
                                        current_time_str = datetime.now(ZoneInfo(ZONE)).strftime("%Y-%m-%d %H:%M:%S")
                                        raw_result = {
                                            "status": "success",
                                            "current_time": current_time_str,
                                            "time_zone": ZONE
                                        }
                                        search_result = json.dumps(raw_result, ensure_ascii=False)
                                    except Exception as e:
                                        search_result = json.dumps({"error": str(e)})
                                    
                                    conversation_history.append({
                                        "tool_call_id": tool_call.id, 
                                        "role": "tool",
                                        "name": "current_time",
                                        "content": search_result,
                                    })
                            
                            try:
                                final_response = groq.chat.completions.create(
                                    messages=conversation_history,
                                    model="llama-3.3-70b-versatile",
                                    temperature=0.7,
                                    tools=tools_schema,
                                    tool_choice="none"
                                )
                                return final_response.choices[0].message.content or ""
                            except BadRequestError as e:
                                print(f"BadRequestError Step 2: {e}")
                                return "Error generating final response."
                        else:
                            return response_message.content or ""
                    
                    try:
                        response_text = await bot.loop.run_in_executor(thread_pool, ask_groq)
                        
                        if not response_text:
                            print("Response text is empty or None.")
                            return

                        conversation_history.append({"role": "assistant", "content": response_text})
                        if len(conversation_history) > 15: 
                            del conversation_history[1:4]

                        vc = guild.voice_client
                        if vc:
                            clean_tts = response_text.replace("**", "").replace("*", "")
                            await speak_response(vc, clean_tts)

                        if LOGGING:
                            embed = discord.Embed(description=response_text, color=discord.Color.red())
                            embed.set_author(name=bot.user.name, icon_url=bot.user.avatar.url if member else None)
                            await channel.send(embed=embed)
                        
                    except Exception as e:
                        print(f"LLM Error: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"No permission in {channel.name}")
    except Exception as e:
        print(f"Transcription error: {e}")


def play_keep_alive(vc):
    """
    Plays a silent audio stream to keep the bot's audio connection alive.

    Args:
        vc: The voice client.
    """
    if not vc.is_connected():
        return
    
    if IS_SPEAKING: 
        return
    
    if vc.is_playing(): 
        return

    source = discord.FFmpegPCMAudio(
        "https://github.com/anars/blank-audio/raw/master/1-minute-of-silence.mp3",
        before_options="-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
    )

    try:
        vc.play(source, after=lambda e: play_keep_alive(vc))
    except discord.ClientException:
        pass
    except Exception as e:
        print(f"Keep-Alive Error: {e}")


@bot.slash_command(name="join")
async def join(ctx):
    """
    Bot joins the user's voice channel after the /join command.

    Args:
        ctx: The command context.
    """
    if not ctx.author.voice:
        return await ctx.respond("Join the voice channel first!")
    
    dest = ctx.author.voice.channel
    vc = ctx.guild.voice_client

    if vc:
        if vc.channel.id != dest.id:
            await vc.move_to(dest)
    else:
        vc = await dest.connect()
    
    try:
        play_keep_alive(vc)
    except Exception as e:
        print(f"FFmpeg not found - Keep-Alive inactive. The bot may go deaf after a minute of silence. (Error: {e})")

    await ctx.respond(f"Connected to **{dest.name}**.")
    if not vc.recording:
        vc.start_recording(
            AutoCutSink(ctx.channel), 
            finished_callback, 
            ctx.channel
        )

    bot_controllers[ctx.guild.id] = ctx.author.id


@bot.slash_command(name="stop")
async def stop(ctx):
    """
    Bot disconnects from the voice channel after the /stop command.
    
    Args:
        ctx: The command context.
    """
    if ctx.guild.voice_client:
        await ctx.guild.voice_client.disconnect()
        await ctx.respond("Disconnected.")
        print("Bot disconnected.")


@bot.event
async def on_voice_state_update(member, before, after):
    """
    Handles voice state updates to manage bot disconnection or movement.
    If the user leaves the channel, the bot will disconnect.
    If the user changes the channel, the bot will follow.

    Args:
        member: The member whose voice state has changed.
        before: The previous voice state.
        after: The new voice state.
    """
    vc = member.guild.voice_client
    if not vc: 
        return
    
    controller_id = bot_controllers.get(member.guild.id)

    if member.id != controller_id: 
        return

    if len(vc.channel.members) == 1:
        await vc.disconnect()
        return

    if before.channel and before.channel.id == vc.channel.id:
        if after.channel:
            if after.channel.id != before.channel.id:
                await vc.move_to(after.channel)
        else:
            await vc.disconnect()
            if member.guild.id in bot_controllers:
                del bot_controllers[member.guild.id]


@bot.event
async def on_ready():
    """
    Function called when the bot is ready.
    """
    print(f"{bot.user} is online!")
    if not check_silence_task.is_running(): check_silence_task.start()


bot.run(BOT_TOKEN)