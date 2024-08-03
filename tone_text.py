from utility import analyze_sentiment, transcibe_audio, custom_round
import openai
from utility import get_api_key
openai.api_key = get_api_key('api_key.txt')

def generate_tonal_feedback(sentences, audio_tone_values):
    t = " ".join(sentences)
    
    prompt = t + f"\nDetermine the percent importance of each sentence. DO NOT GIVE ANY OTHER EXTRA FEEDBACK OR EXPLANATION FOR YOUR CHOICES! Do not add percent symbols. There are {len(sentences)} sentences and give {len(sentences)} responses. Put it into the following format: [Percent imortance of the first sentence] | [Percent importance of the second sentence] | [Percent importance of the third sentence] (and so on and so forth)."
        
    def chat_with_gpt(prompt):
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a english teacher who determines which sentence holds the most importance."},
                {"role": "user", "content": prompt},
            ]
        )

        return response.choices[0].message.content
    
    response = chat_with_gpt(prompt)

    importance = response.split(' | ')
    importance = [int(i) for i in importance]
    total_importance = sum(importance)
    scale_factor = 100 / total_importance
    scaled_numbers = [x * scale_factor for x in importance]

    sentiment = [custom_round(analyze_sentiment(s)['compound']) for s in sentences]
    
    score = 0
    for i in range(len(sentences)):
        scale = 1 / (abs(sentiment[i] - audio_tone_values[i]) + 1)
        score += scale * scaled_numbers[i]
    
    return score, importance, sentiment
