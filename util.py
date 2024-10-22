import os
from pydub import AudioSegment
import openai
import ssl
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from cryptography.fernet import Fernet

ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('vader_lexicon')
ssl._create_default_https_context = ssl.create_default_context

def custom_round(value, threshold=0.1):
    if value > threshold:
        return 1
    elif value < -threshold:
        return -1
    else:
        return 0

def generate_key():
    key = Fernet.generate_key()
    with open("secret.key", "wb") as key_file:
        key_file.write(key)
    return key

def load_key():
    try:
        return open("secret.key", "rb").read()
    except FileNotFoundError:
        raise FileNotFoundError("Encryption key not found. Please generate a new key.")

def encrypt_api_key(api_key):
    key = load_key()
    fernet = Fernet(key)
    encrypted_key = fernet.encrypt(api_key.encode())
    return encrypted_key

def write_encrypted_api_key(api_key, file_path):
    encrypted_key = encrypt_api_key(api_key)
    
    with open(file_path, "wb") as file:
        file.write(encrypted_key)

    print(f"API key has been encrypted and saved to {file_path}")

def get_api_key(file_path):
    try:
        with open(file_path, "rb") as file:
            encrypted_key = file.read()

        if not encrypted_key:
            raise ValueError("API key file is empty")

        key = load_key()
        fernet = Fernet(key)
        api_key = fernet.decrypt(encrypted_key).decode()
        return api_key

    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at: {file_path}")
    except Exception as e:
        raise e

import streamlit as st
openai.api_key = st.secrets["openai"]["api_key"]

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

def split_audio_file(file_path, max_size_mb=25):
    max_size_bytes = max_size_mb * 1024 * 1024
    file_size = os.path.getsize(file_path)

    audio = AudioSegment.from_file(file_path)
    
    base_name = os.path.basename(file_path).rsplit('.', 1)[0]
    
    output_folder = f"split_audio/{base_name}_parts"
    os.makedirs(output_folder, exist_ok=True)

    if file_size <= max_size_bytes:
        chunk_file_path = os.path.join(output_folder, f"{base_name}_part1.wav")
        audio.export(chunk_file_path, format="wav")
        print(f"File is under {max_size_mb} MB. Exported as {chunk_file_path}")
        return output_folder, [chunk_file_path]

    num_chunks = file_size // max_size_bytes + 1
    chunk_duration_ms = len(audio) // num_chunks
    
    audio_chunks = []
    for i in range(num_chunks):
        start_time = i * chunk_duration_ms
        end_time = (i + 1) * chunk_duration_ms
        chunk = audio[start_time:end_time]
        chunk_file_path = os.path.join(output_folder, f"{base_name}_part{i+1}.wav")
        chunk.export(chunk_file_path, format="wav")
        audio_chunks.append(chunk_file_path)
        print(f"Chunk {i+1} exported as {chunk_file_path}")

    return output_folder, audio_chunks

class Analysis:
    def __init__(self, audio_path, speech_type='impromptu', text=None) -> None:
        self.audio_path = audio_path
        self.speech_type = speech_type
        base_name = os.path.basename(audio_path).rsplit('.', 1)[0]
        self.dir = f'split_audio/{base_name}_parts'

        if not os.path.isdir(self.dir):
            os.makedirs(self.dir, exist_ok=True)
            _, self.audio_paths = split_audio_file(self.audio_path, max_size_mb=24)
        else:
            self.audio_paths = [os.path.join(self.dir, f) for f in os.listdir(self.dir)]
            self.audio_paths = [os.path.relpath(path) for path in self.audio_paths]
        
        if text:
            self.transcription_time_text, self.transcription_text, self.transcription_sentence, self.transcription_time_sentence = text
        else:
            self.transcription_time_text, self.transcription_text, self.transcription_sentence, self.transcription_time_sentence = self.transcribe_audio()
    
    def transcribe_audio(self):
        transcriptions_with_time = []
        cumulative_time_offset = 0
        full_transcription_text = ""
        sentence_transcriptions = []
        sentence_times = []

        for audio_file in self.audio_paths:
            with open(audio_file, 'rb') as file:
                transcription = openai.audio.transcriptions.create(
                    file=file,
                    model='whisper-1',
                    response_format='verbose_json',
                    timestamp_granularities=['word']
                )
                
                words = transcription.words
                chunk_text = transcription.text.strip()
                full_transcription_text += chunk_text + " "
                
                for word in words:
                    print(word)
                    word['start'] += cumulative_time_offset
                    word['end'] += cumulative_time_offset

                chunk_duration = AudioSegment.from_file(audio_file).duration_seconds
                cumulative_time_offset += chunk_duration

                transcriptions_with_time.extend(words)

                sentences = re.split(r'(?<=[.!?]) +', chunk_text)
                current_index = 0

                for sentence in sentences:
                    sentence_start_time = None
                    sentence_end_time = None
                    sentence_length = len(sentence.split())
                    matched_words = 0
                    sentence_text = ''

                    for word_info in words[current_index:]:
                        word = word_info['word']
                        
                        if matched_words == 0:
                            sentence_start_time = word_info['start']
                        
                        sentence_text += ' ' + word
                        matched_words += 1
                        
                        if matched_words == sentence_length:
                            sentence_end_time = word_info['end']
                            current_index += matched_words
                            break
                    
                    if sentence_start_time is not None and sentence_end_time is not None:
                        sentence_times.append({
                            'text': sentence.strip(),
                            'start': sentence_start_time,
                            'end': sentence_end_time
                        })
                        sentence_transcriptions.append(sentence.strip())

        full_transcription_text = full_transcription_text.strip()

        return transcriptions_with_time, full_transcription_text, sentence_transcriptions, sentence_times
