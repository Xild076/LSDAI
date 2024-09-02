from util import Analysis, get_api_key
import openai
openai.api_key = get_api_key('api_key.encrypted')


class Content(Analysis):
    def __init__(self, audio_path, speech_type='impromptu', text=None) -> None:
         super().__init__(audio_path, speech_type, text)
    
    def get_content_feedback(self):
        while True:
            try:
                prompt = self.transcription_text + "\nGive general feedback for the content of this speech. Then give a score out of 100. Put it into the following format: {feedback} | {score}. The score must be just its value, not x/100, nor with a period at the end."
                
                def chat_with_gpt(prompt):
                    response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a speech coach who gives feedback for a student's content in their speeches."},
                            {"role": "user", "content": prompt},
                        ]
                    )
                    return response.choices[0].message.content
                
                result = chat_with_gpt(prompt).split(' | ')
                proper_response = result[0]
                score_fbck = int(result[1])
                break
            except:
                pass

        return proper_response, score_fbck

print("Content Class Loaded")