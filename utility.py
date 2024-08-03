import nltk.downloader
import openai
from pydub import AudioSegment
import json
from enum import Enum
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import ssl
import re

ssl._create_default_https_context = ssl._create_unverified_context
from utility import get_api_key
openai.api_key = get_api_key('api_key.txt')
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('vader_lexicon')
ssl._create_default_https_context = ssl.create_default_context

def get_api_key(file_path):
    try:
        with open(file_path, 'r') as file:
            api_key = file.read().strip()
            if not api_key:
                raise ValueError("API key file is empty")
            return api_key
    except FileNotFoundError:
        raise FileNotFoundError(f"API key file not found at: {file_path}")
    except Exception as e:
        raise e

def transcibe_audio(audio_path, creation_type):
    with open(audio_path, 'rb') as audio_file:
        if creation_type == 'text':
            transcription = openai.audio.transcriptions.create(
                file=audio_file,
                model='whisper-1',
                response_format='text'
            )
            return transcription.strip()
        elif creation_type == 'verbose_json':
            transcription = openai.audio.transcriptions.create(
                file=audio_file,
                model='whisper-1',
                response_format='verbose_json',
                timestamp_granularities=['word']
            )
            return transcription.words
        elif creation_type == 'sentence':
            transcription = openai.audio.transcriptions.create(
                file=audio_file,
                model='whisper-1',
                response_format='text'
            )
            sentences = transcription.strip().split('. ')
            sentences = [sentence.rstrip() + '.' if not sentence.endswith('.') else sentence for sentence in sentences]
            return [sentence for sentence in sentences if sentence != '.']
        elif creation_type == 'sentence_verbose_json':
            transcription = openai.audio.transcriptions.create(
                file=audio_file,
                model='whisper-1',
                response_format='verbose_json',
                timestamp_granularities=['word']
            )
            
            words = transcription.words
            text = transcription.text
            
            sentences = re.split(r'(?<=[.!?]) +', text)
            
            sentence_times = []
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
            
            return sentence_times


def write_jsonl_file(filename, data, append=False):
    mode = 'a' if append else 'w'
    with open(filename, mode) as f:
        for pair in data:
            user_content, assistant_content = pair
            entry = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": assistant_content}
                ]
            }
            f.write(json.dumps(entry) + '\n')

def strip_silence(audio_path, output: tuple, silence_threshold=-100, silence_len=500):
    audio = AudioSegment.from_file(audio_path)
    filtered_audio = audio.strip_silence(silence_len=silence_len, silence_thresh=silence_threshold)
    
    output_vals = []
    for o in output:
        if o == AudioOutput.TEXT:
            output_vals.append(transcibe_audio(audio_path, 'text'))
        if o == AudioOutput.LENGTH:
            output_vals.append(len(filtered_audio) / 1000)
    return tuple(output_vals)

class AudioOutput(Enum):
    TEXT = 0
    LENGTH = 1

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(text)
    return sentiment_scores

def custom_round(value, threshold=0.1):
    if value > threshold:
        return 1
    elif value < -threshold:
        return -1
    else:
        return 0