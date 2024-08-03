from utility import strip_silence, AudioOutput
import math

def word_per_minute(audio_path, silence_threshold=-100, silence_len=500):
    def bell_score_calc(wpm):
        div = (wpm - 125) / 20
        sech_func = (2*math.e**(div)) / (math.e**(2*div) + 1)
        return round(100 * sech_func)
    text, time_len = strip_silence(audio_path, (AudioOutput.TEXT, AudioOutput.LENGTH), silence_threshold, silence_len)

    wpm = len(text.split(' ')) / (time_len / 60)
    
    return wpm, bell_score_calc(wpm)