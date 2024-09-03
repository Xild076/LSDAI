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
    
    def calculate_word_volumes(self, audio_path, words_timestamps):
        y, sr = librosa.load(audio_path, sr=None)
        
        word_volumes = []
        for word_info in words_timestamps:
            word_start = word_info['start']
            word_end = word_info['end']
            start_sample = librosa.time_to_samples(word_start, sr=sr)
            end_sample = librosa.time_to_samples(word_end, sr=sr)
            word_audio = y[start_sample:end_sample]
            if len(word_audio) > 0:
                word_volume = np.mean(np.abs(word_audio))
                word_volumes.append(word_volume)
            else:
                word_volumes.append(0)
        
        return word_volumes
    
    def find_loud_words(self, words_timestamps, relative_window_percentage=0.1, loud_threshold_factor=1.5):
        word_volumes = self.calculate_word_volumes(self.audio_path, words_timestamps)
        total_words = len(word_volumes)
        window_size = max(1, int(total_words * relative_window_percentage))
        loud_words_global = []
        loud_words_surrounding = []
        global_avg_volume = np.mean(word_volumes)

        for i, word_info in enumerate(words_timestamps):
            start_index = max(0, i - window_size)
            end_index = min(total_words, i + window_size + 1)
            surrounding_volumes = np.array(word_volumes[start_index:end_index])
            surrounding_volumes = surrounding_volumes[surrounding_volumes != word_volumes[i]]
            
            if len(surrounding_volumes) > 0:
                avg_surrounding_volume = np.mean(surrounding_volumes)
                if word_volumes[i] > avg_surrounding_volume * loud_threshold_factor:
                    loud_words_surrounding.append((word_info['word'], word_info['start'], word_info['end']))
                if word_volumes[i] > global_avg_volume * loud_threshold_factor:
                    loud_words_global.append((word_info['word'], word_info['start'], word_info['end']))
        
        combined_loud_words = list(set(loud_words_global + loud_words_surrounding))
        combined_loud_words.sort(key=lambda word: -word_volumes[words_timestamps.index({'word': word[0], 'start': word[1], 'end': word[2]})])
        max_loud_words = total_words // 5
        selected_loud_words = combined_loud_words[:max_loud_words]
        
        return selected_loud_words
    
    def emphasize_loud_words_in_text(self, loud_words):
        emphasized_text = self.transcription_text

        loud_words = sorted(loud_words, key=lambda x: len(x[0]), reverse=True)

        for word_info in loud_words:
            word = re.escape(word_info[0])
            emphasized_text = re.sub(rf'\b{word}\b', f"***{word_info[0]}***", emphasized_text)
            emphasized_text = emphasized_text.replace('$', 'S')

        emphasized_text = re.sub(r'***( ***)+***', '***', emphasized_text)
        
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
                            {"role": "system", "content": "You are a speech coach who give feedback for a student's use of emphasis in their {0} speeches.".format(self.speech_type)},
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