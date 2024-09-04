import openai
import numpy as np
import librosa
from util import get_api_key, Analysis
import re
import streamlit as st

openai.api_key = st.secrets["openai"]["api_key"]

class Emphasis(Analysis):
    def __init__(self, audio_path, speech_type='impromptu', text=None) -> None:
        super().__init__(audio_path, speech_type, text)

    def calculate_word_features(self, audio_path, words_timestamps):
        y, sr = librosa.load(audio_path, sr=None)
        
        word_features = []
        for word_info in words_timestamps:
            word_start = word_info['start']
            word_end = word_info['end']
            start_sample = librosa.time_to_samples(word_start, sr=sr)
            end_sample = librosa.time_to_samples(word_end, sr=sr)
            word_audio = y[start_sample:end_sample]
            
            if len(word_audio) > 0:
                word_volume = np.mean(np.abs(word_audio))
                word_energy = np.mean(word_audio**2)
                word_pitch = np.mean(librosa.yin(word_audio, fmin=75, fmax=300))
                
                word_features.append({
                    'volume': word_volume,
                    'energy': word_energy,
                    'pitch': word_pitch
                })
            else:
                word_features.append({'volume': 0, 'energy': 0, 'pitch': 0})
        
        return word_features

    def find_loud_words(self, words_timestamps, relative_window_percentage=0.1, loud_threshold_factor=1.3, pitch_threshold=1.1):
        word_features = self.calculate_word_features(self.audio_path, words_timestamps)
        total_words = len(word_features)
        window_size = max(1, int(total_words * relative_window_percentage))
        loud_words = []
        global_avg_volume = np.mean([f['volume'] for f in word_features])

        for i, word_info in enumerate(words_timestamps):
            start_index = max(0, i - window_size)
            end_index = min(total_words, i + window_size + 1)
            surrounding_features = word_features[start_index:end_index]
            surrounding_volumes = np.array([f['volume'] for f in surrounding_features if f['volume'] != word_features[i]['volume']])
            surrounding_pitches = np.array([f['pitch'] for f in surrounding_features if f['pitch'] != word_features[i]['pitch']])

            if len(surrounding_volumes) > 0 and len(surrounding_pitches) > 0:
                avg_surrounding_volume = np.mean(surrounding_volumes)
                avg_surrounding_pitch = np.mean(surrounding_pitches)

                if (word_features[i]['volume'] > avg_surrounding_volume * loud_threshold_factor or
                    word_features[i]['volume'] > global_avg_volume * loud_threshold_factor) and \
                    word_features[i]['pitch'] > avg_surrounding_pitch * pitch_threshold:
                    loud_words.append((word_info['word'], word_info['start'], word_info['end']))
        
        return loud_words

    def emphasize_loud_words_in_text(self, loud_words):
        emphasized_text = self.transcription_text
        loud_words = sorted(loud_words, key=lambda x: len(x[0]), reverse=True)

        for word_info in loud_words:
            word = re.escape(word_info[0])
            emphasized_text = re.sub(rf'\b{word}\b', f"_{word_info[0]}_", emphasized_text)
            emphasized_text = emphasized_text.replace('$', 'S')

        emphasized_text = re.sub(r'_( _)+_', '_', emphasized_text)
        
        return emphasized_text.strip()

    def locate_emphasized_words(self):
        words_timestamps = self.transcription_time_text
        loud_words = self.find_loud_words(words_timestamps)
        emphasized_text = self.emphasize_loud_words_in_text(loud_words)
        return emphasized_text

    def get_emphasis_feedback(self):
        while True:
            try:
                emph_text = self.locate_emphasized_words()
                
                prompt = emph_text + "\nIs emphasis in the passage done correctly? If not, please give feedback. Respond as simply as possible and only focus on the emphasis. Then give the emphasis a score out of 100. Put it into the following format: {feedback} | {score}. The score must be just its value, not x/100, nor with a period at the end."
                    
                def chat_with_gpt(prompt):
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a speech coach who gives feedback for a student's use of emphasis in their {0} speeches.".format(self.speech_type)},
                            {"role": "user", "content": prompt},
                        ]
                    )

                    return response.choices[0].message.content
                
                response = chat_with_gpt(prompt)
                part_response = response.split(' ')
                proper_response = ' '.join(part_response[:(len(part_response) - 2)])
                score = int(part_response[-1])
                break
            except:
                pass
        return emph_text, proper_response, score

print("Emphasis Class Loaded")