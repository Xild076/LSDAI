from util import Analysis
import math


class Speed(Analysis):
    def __init__(self, audio_path, speech_type='impromptu', text=None) -> None:
        super().__init__(audio_path, speech_type, text)
    
    def get_wpm_feedback(self):
        start_time = self.transcription_time_text[0].start
        end_time = self.transcription_time_text[-1].end
        wpm = len(self.transcription_time_text) / (end_time - start_time) * 60

        def bell_score_calc(wpm):
            div = (wpm - 150) / 30
            sech_func = (2*math.e**(div)) / (math.e**(2*div) + 1)
            return round(100 * sech_func)
        
        return wpm, bell_score_calc(wpm)

print("Speed Class Loaded")