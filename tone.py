from tone_a import get_audio_emotion
from utility import transcibe_audio
from tone_text import generate_tonal_feedback

def determine_emotion_value(emotion):
    val = 0
    _, emotion_type = emotion.split('_', 1)
    if emotion_type in ['sad', 'mad', 'angry', 'disgust', 'fear']:
        val = -1
    elif emotion_type == 'neutral':
        val = 0
    elif emotion_type in ['happy', 'surprise']:
        val = 1
    elif emotion_type == 'Unknown':
        val = 0
    return val

def generate_emotion_audio_determination(audio_path):
    audio_emotion = get_audio_emotion(audio_path)
    tone = []
    for ae in audio_emotion:
        tone.append(determine_emotion_value(ae['emotion'][0]))
    return tone

def get_full_tonal_feedback(audio_path):
    tt = transcibe_audio(audio_path, 'sentence')
    audio_tone_values = generate_emotion_audio_determination(audio_path)
    score, importance, sentiment = generate_tonal_feedback(tt, audio_tone_values)
    return score, importance, sentiment, audio_tone_values
