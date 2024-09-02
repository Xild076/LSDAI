from speed import Speed
from tone import Tone
from content import Content
from emphasis import Emphasis

class SpeechAnalysis:
    def __init__(self, audio_path, speech_type) -> None:
        self.content = Content(audio_path, speech_type)
        self.emphasis = Emphasis(audio_path, speech_type, (self.content.transcription_time_text, self.content.transcription_text, self.content.transcription_sentence, self.content.transcription_time_sentence))
        self.tone = Tone(audio_path, speech_type, (self.content.transcription_time_text, self.content.transcription_text, self.content.transcription_sentence, self.content.transcription_time_sentence))
        self.speed = Speed(audio_path, speech_type, (self.content.transcription_time_text, self.content.transcription_text, self.content.transcription_sentence, self.content.transcription_time_sentence))

        self.speech_type = speech_type
    
    def analyse(self, configs=None):
        feedback = {}
        if not configs:
            if self.speech_type == 'impromptu' or 'extempt':
                configs = [1, 1, 1, 1]
            else:
                configs = [0, 1, 1, 1]
        if configs[0] == 1:
            feedback['content'] = self.content.get_content_feedback()
        if configs[1] == 1:
            feedback['emphasis'] = self.emphasis.get_emphasis_feedback()
        if configs[2] == 1:
            feedback['tone'] = self.tone.get_tonal_feedback()
        if configs[3] == 1:
            feedback['speed'] = self.speed.get_wpm_feedback()

        return feedback, configs
