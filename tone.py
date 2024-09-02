from util import analyze_sentiment, Analysis
import openai
from util import get_api_key, custom_round
openai.api_key = get_api_key('api_key.txt')
import os
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import zipfile
import numpy as np
import pickle
import IPython.display as ipd
from tqdm import tqdm
import gdown
from IPython.display import display
import sys
import warnings
from pydub import AudioSegment
import resampy

if not sys.warnoptions:
    warnings.simplefilter("ignore")
device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')


class Tone(Analysis):
    def __init__(self, audio_path, speech_type='impromptu', text=None) -> None:
        super().__init__(audio_path, speech_type, text)
        print("Initialized!")
        self.tone_train = ToneTrain(sentence_time=self.transcription_time_sentence)
        if not os.path.isfile('tone_category/emotion_recog_model.pth'):
            print("Loading, loading...")
            self.tone_train.train()
    
    def determine_emotion_value(self, emotion):
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
    
    def categorize_values(self, values):
        sorted_values = sorted(values)
        n = len(values)
        
        low_size = n // 2
        medium_size = n // 3
        high_size = n - low_size - medium_size
        
        categorized = []
        
        for i, value in enumerate(sorted_values):
            if i < low_size:
                categorized.append((value, 'Low'))
            elif i < low_size + medium_size:
                categorized.append((value, 'Medium'))
            else:
                categorized.append((value, 'High'))
        
        return categorized

    
    def generate_emotion_audio_determination(self):
        audio_emotion = self.tone_train.get_audio_emotion(self.audio_path)
        tone = []
        for ae in audio_emotion:
            tone.append(self.determine_emotion_value(ae['emotion'][0]))
        return tone

    def get_tonal_feedback(self):
        while True:
            try:
                tt = self.transcription_sentence
                audio_tone_values = self.generate_emotion_audio_determination()
                score_calc, importance, sentiment = self.generate_tonal_feedback(tt, audio_tone_values)
                text_sentiment = ""
                for t, i, s, a in zip(tt, importance, sentiment, audio_tone_values):
                    st = 'Neutral'
                    if s == -1:
                        st = 'Negative'
                    if s == 1:
                        st = 'Positive'
                    at = 'Neutral'
                    if a == -1:
                        at = 'Negative'
                    if a == 1:
                        at = 'Positive'
                    text_sentiment += '**Sentence:** _' + t + '_ | **Importance:** ' + str(i) + ' | **Text Sentiment:** _' + st + '_ | **Audio Sentiment:** _' + at + '_\n\n '
                
                def chat_with_gpt(prompt):
                    response = openai.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a speech teacher who gives feedback on the tone of the student and how they can better use their tone."},
                            {"role": "user", "content": prompt},
                        ]
                    )

                    return response.choices[0].message.content
                
                prompt = text_sentiment + '\n\nPlease give feedback on how well done the tone usage is done and feedback on how to make better use of tone. Dont do it sentence by sentence, just do a general tone feedback. Then give the tonal usage a score out of 100. Put it into the following format: {feedback} | {score}. The score must be just its value, not x/100, nor with a period at the end.'
                response = chat_with_gpt(prompt)
                part_response = response.split(' ')
                proper_response = ' '.join(part_response[:(len(part_response) - 2)])
                score_gpt = int(part_response[-1])
                total_score = (score_gpt + score_calc) / 2
                break
            except:
                pass
        return text_sentiment, proper_response, total_score
    
    def generate_tonal_feedback(self, sentences, audio_tone_values):
        while True:
            try:
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
                break
            except:
                pass
        return score, importance, sentiment


class ToneTrain:
    def __init__(self, sentence_time) -> None:
        self.CREMA = 'kaggle/CREMA/'
        self.SAVEE = 'kaggle/SAVEE/'
        self.TESS = 'kaggle/TESS/'
        self.RAVDESS = 'kaggle/RAVDESS/'
        self.sentence_time = sentence_time


        self.config = {
            'batch_size': 16,
            'num_workers': 2,
            'learning_rate': 1e-5,
            'num_epochs': 500,
            'train_size': 0.98,
            'mfcc_duration': 2.5,
            'mfcc_sr': 44100,
            'mfcc_offset': 0.5,
            'n_mfcc': 13
        }

    def load_CREMA_data(self, in_path, info=False):
        dir_list = os.listdir(in_path)
        dir_list.sort()

        if info:
            print(dir_list[0:5])

        gender = []
        emotion = []
        path = []
        female = [1002, 1003, 1004, 1006, 1007, 1008, 1009, 1010, 1012, 1013, 1018, 1020, 1021, 1024, 1025, 1028, 1029, 1030, 1037, 1043, 1046, 1047, 1049,
                1052, 1053, 1054, 1055, 1056, 1058, 1060, 1061, 1063, 1072, 1073, 1074, 1075, 1076, 1078, 1079, 1082, 1084, 1089, 1091]
        
        for i in dir_list:
            part = i.split('_')
            if int(part[0]) in female:
                temp = 'female'
            else:
                temp = 'male'
            gender.append(temp)
            if part[2] == 'SAD' and temp == 'male':
                emotion.append('male_sad')
            elif part[2] == 'ANG' and temp == 'male':
                emotion.append('male_angry')
            elif part[2] == 'DIS' and temp == 'male':
                emotion.append('male_disgust')
            elif part[2] == 'FEA' and temp == 'male':
                emotion.append('male_fear')
            elif part[2] == 'HAP' and temp == 'male':
                emotion.append('male_happy')
            elif part[2] == 'NEU' and temp == 'male':
                emotion.append('male_neutral')
            elif part[2] == 'SAD' and temp == 'female':
                emotion.append('female_sad')
            elif part[2] == 'ANG' and temp == 'female':
                emotion.append('female_angry')
            elif part[2] == 'DIS' and temp == 'female':
                emotion.append('female_disgust')
            elif part[2] == 'FEA' and temp == 'female':
                emotion.append('female_fear')
            elif part[2] == 'HAP' and temp == 'female':
                emotion.append('female_happy')
            elif part[2] == 'NEU' and temp == 'female':
                emotion.append('female_neutral')
            else:
                emotion.append('Unknown')
            path.append(in_path + i)
        
        CREMA_df = pd.DataFrame(emotion, columns=['labels'])
        CREMA_df['source'] = 'CREMA'
        CREMA_df = pd.concat([CREMA_df, pd.DataFrame(path, columns=['path'])], axis=1)

        if info:
            print(CREMA_df.labels.value_counts())
            fname = in_path + '1012_IEO_HAP_HI.wav'
            data, sampling_rate = librosa.load(fname)
            plt.figure(figsize=(15, 5))
            librosa.display.waveshow(data, sr=sampling_rate)
            plt.show()
        
        return CREMA_df

    def load_SAVEE_data(self, in_path, info=False):
        dir_list = os.listdir(in_path)
        emotion=[]
        path = []
        for i in dir_list:
            if i[-8:-6]=='_a':
                emotion.append('male_angry')
            elif i[-8:-6]=='_d':
                emotion.append('male_disgust')
            elif i[-8:-6]=='_f':
                emotion.append('male_fear')
            elif i[-8:-6]=='_h':
                emotion.append('male_happy')
            elif i[-8:-6]=='_n':
                emotion.append('male_neutral')
            elif i[-8:-6]=='sa':
                emotion.append('male_sad')
            elif i[-8:-6]=='su':
                emotion.append('male_surprise')
            else:
                emotion.append('male_error') 
            path.append(in_path + i)
        
        SAVEE_df = pd.DataFrame(emotion, columns = ['labels'])
        SAVEE_df['source'] = 'SAVEE'
        SAVEE_df = pd.concat([SAVEE_df, pd.DataFrame(path, columns = ['path'])], axis = 1)
        return SAVEE_df

    def load_RAVDESS_data(self, in_path, info=False):
        dir_list = os.listdir(in_path)
        dir_list.sort()

        emotion = []
        gender = []
        path = []
        for i in dir_list:
            fname = os.listdir(in_path + i)
            for f in fname:
                part = f.split('.')[0].split('-')
                emotion.append(int(part[2]))
                temp = int(part[6])
                if temp % 2 == 0:
                    temp = "female"
                else:
                    temp = "male"
                gender.append(temp)
                path.append(in_path + i + '/' + f)

        RAV_df = pd.DataFrame(emotion)
        RAV_df = RAV_df.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'})
        RAV_df = pd.concat([pd.DataFrame(gender), RAV_df], axis=1)
        RAV_df.columns = ['gender', 'emotion']
        RAV_df['labels'] = RAV_df.gender + '_' + RAV_df.emotion
        RAV_df['source'] = 'RAVDESS'  
        RAV_df = pd.concat([RAV_df, pd.DataFrame(path, columns=['path'])], axis=1)
        RAV_df = RAV_df.drop(['gender', 'emotion'], axis=1)
        return RAV_df

    def load_TESS_data(self, in_path, info=False):
        dir_list = os.listdir(in_path)
        dir_list.sort()
        path = []
        emotion = []

        for i in dir_list:
            dir_path = os.path.join(in_path, i)
            if not os.path.isdir(dir_path):
                continue
            fname = os.listdir(dir_path)
            for f in fname:
                if i == 'OAF_angry' or i == 'YAF_angry':
                    emotion.append('female_angry')
                elif i == 'OAF_disgust' or i == 'YAF_disgust':
                    emotion.append('female_disgust')
                elif i == 'OAF_Fear' or i == 'YAF_fear':
                    emotion.append('female_fear')
                elif i == 'OAF_happy' or i == 'YAF_happy':
                    emotion.append('female_happy')
                elif i == 'OAF_neutral' or i == 'YAF_neutral':
                    emotion.append('female_neutral')                                
                elif i == 'OAF_Pleasant_surprise' or i == 'YAF_pleasant_surprised':
                    emotion.append('female_surprise')               
                elif i == 'OAF_Sad' or i == 'YAF_sad':
                    emotion.append('female_sad')
                else:
                    emotion.append('Unknown')
                path.append(in_path + i + "/" + f)

        TESS_df = pd.DataFrame(emotion, columns = ['labels'])
        TESS_df['source'] = 'TESS'
        TESS_df = pd.concat([TESS_df, pd.DataFrame(path, columns = ['path'])], axis=1)
        return TESS_df

    def extract_features(self, in_path, out_path, info=False):
        ref = pd.read_csv(in_path)
        ref.head()

        if info:
            print(ref)
        
        df = pd.DataFrame(columns=['feature'])
        counter = 0
        for index, path in tqdm(enumerate(ref.path), total=len(ref.path)):
            try:
                X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=self.config['mfcc_duration'], sr=self.config['mfcc_sr'], offset=self.config['mfcc_offset'])
                mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=self.config['n_mfcc']), axis=0)
                df.loc[counter] = [mfccs]
                counter += 1
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        df = pd.concat([ref, pd.DataFrame(df['feature'].values.tolist())], axis=1)
        df = df.fillna(0)
        df.to_csv(out_path, index=False)
    
    def move_to_device(self, dataloader, device):
        data_on_device = []
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            data_on_device.append((inputs, labels))
        return data_on_device
    
    def train_model(self, model, criterion, optimizer, train_data, num_epochs):
        model.train()
        train_losses = []
        train_accuracies = []
        total_samples = sum(inputs.size(0) for inputs, _ in train_data)

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            for inputs, labels in tqdm(train_data, desc=f'Epoch {epoch + 1}/{num_epochs}'):
                optimizer.zero_grad()
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs.data, 1)
                _, actual = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == actual).sum().item()
            
            epoch_loss = running_loss / total_samples
            epoch_acc = 100 * correct / total
            train_losses.append(epoch_loss)
            train_accuracies.append(epoch_acc)
            print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')

        return model, train_losses, train_accuracies
    
    def validate_model(self, model, val_data, dataset):
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_data):
                outputs = model.forward(inputs)
                _, predicted = torch.max(outputs.data, 1)
                _, actual = torch.max(labels.data, 1)
                total += labels.size(0)
                correct += (predicted == actual).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(actual.cpu().numpy())

                if i < 5:
                    audio_path = dataset.data.iloc[i]['path']
                    print(f'Playing: {audio_path}')
                    display(ipd.Audio(audio_path))
                    print(f'Predicted: {dataset.lb.inverse_transform([predicted.cpu().numpy()[0]])[0]}')
                    print(f'Actual: {dataset.lb.inverse_transform([actual.cpu().numpy()[0]])[0]}')
                    print()
        
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')

        return accuracy
    
    def plot_training_progress(self, train_losses, train_accuracies):
        epochs = range(1, len(train_losses) + 1)
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'r', label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accuracies, 'b', label='Training Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.legend()

        plt.show()

    def save_model(self, model, model_path, label_encoder, encoder_path):
        torch.save(model.state_dict(), model_path)
        with open(encoder_path, 'wb') as f:
            pickle.dump(label_encoder, f)

    def load_model(self, model_path, encoder_path, num_classes, device):
        model = AudioModel(num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        return model, label_encoder

    def predict_emotion(self, audio_path, model, label_encoder):
        try:
            model.eval()
            X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=self.config['mfcc_duration'], sr=self.config['mfcc_sr'], offset=self.config['mfcc_offset'])
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=self.config['n_mfcc']), axis=0)
            newdf = pd.DataFrame(data=mfccs).T
            newdf = torch.tensor(newdf.values, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                newpred = model.forward(newdf)
                final = newpred.argmax(axis=1).cpu().numpy()
                final = final.astype(int).flatten()
                final = label_encoder.inverse_transform(final)
        except Exception as ex:
            return ['none_Unknown']
        
        return final
    
    def train(self):
        if not os.path.isfile('data/data_path.csv'):
            print("Loading data paths...")
            cmdf, svdf, rvtf, tstf = self.load_CREMA_data(self.CREMA), self.load_SAVEE_data(self.SAVEE), self.load_RAVDESS_data(self.RAVDESS), self.load_TESS_data(self.TESS)
            df = pd.concat([cmdf, svdf, rvtf, tstf], axis=0)
            df.to_csv("data/data_path.csv", index=False)
            self.extract_features('data/data_path.csv', 'data/train_path.csv')

        print("Loading dataset...")
        dataset = AudioDataset('data/train_path.csv')
        train_size = int((self.config['train_size']) * len(dataset))
        val_size = len(dataset) - train_size
        train_data, val_data = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(train_data, batch_size=self.config['batch_size'], shuffle=True, num_workers=self.config['num_workers'])
        val_dataloader = DataLoader(val_data, batch_size=self.config['batch_size'], shuffle=False, num_workers=self.config['num_workers'])

        train_data_on_device = self.move_to_device(train_dataloader, device)
        val_data_on_device = self.move_to_device(val_dataloader, device)

        print("Loading model...")
        model = AudioModel(dataset.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])

        print("Starting to train...")
        model, train_losses, train_accuracies = self.train_model(model, criterion, optimizer, train_data_on_device, self.config['num_epochs'])

        self.plot_training_progress(train_losses, train_accuracies)

        self.validate_model(model, val_data_on_device, dataset)

        self.save_model(model, 'tone_category/emotion_recog_model.pth', dataset.lb, 'tone_category/label_encoder.pkl')
    
    def get_audio_emotion(self, audio_path):
        dataset = 14
        model_path = 'tone_category/emotion_recog_model.pth'
        encoder_path = 'tone_category/label_encoder.pkl'
        
        loaded_model, loaded_label_encoder = self.load_model(model_path, encoder_path, dataset, device)
        
        audio = AudioSegment.from_file(audio_path)
        sentence_times = self.sentence_time
        
        emotions = []
        for sentence in sentence_times:
            start_ms = sentence['start'] * 1000
            end_ms = sentence['end'] * 1000
            audio_segment = audio[start_ms:end_ms]
            
            segment_path = "temp/temp_segment.wav"
            audio_segment.export(segment_path, format="wav")
            
            emotion = self.predict_emotion(segment_path, loaded_model, loaded_label_encoder)
            emotions.append({
                'text': sentence['text'],
                'start': sentence['start'],
                'end': sentence['end'],
                'emotion': emotion
            })
        
        
        return emotions


class AudioModel(nn.Module):
    def __init__(self, num_classes):
        super(AudioModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 256, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.MaxPool1d(8),
            nn.Dropout(0.25),
            
            nn.Conv1d(256, 128, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(128, 128, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(8),
            nn.Dropout(0.25),
            
            nn.Conv1d(128, 64, kernel_size=8, padding='same'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=8, padding='same'),
            nn.ReLU()
        )
        
        self.flatten_dim = 64 * (192 // 64)
        self.fc = nn.Linear(self.flatten_dim, num_classes)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class AudioDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.X = self.data.drop(['path', 'labels', 'source'], axis=1).values
        self.y = self.data['labels'].values
        self.lb = LabelEncoder()
        self.y = self.lb.fit_transform(self.y)
        self.num_classes = len(np.unique(self.y)) 
        self.y = torch.nn.functional.one_hot(torch.tensor(self.y), num_classes=self.num_classes).float()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]


print("Tone Class Loaded")
