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

from utility import transcibe_audio
import resampy

if not sys.warnoptions:
    warnings.simplefilter("ignore")


CREMA = 'kaggle/cremad/AudioWav/'
SAVEE = 'kaggle/savee/ALL/'
TESS = 'kaggle/tess/TESS Toronto emotional speech set data/TESS Toronto emotional speech set data/'
RAVDESS = 'kaggle/ravdess/audio_speech_actors_01-24/'

device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu')
config = {
    'batch_size': 16,
    'num_workers': 2,
    'learning_rate': 1e-5,
    'num_epochs': 1000,
    'train_size': 0.98,
    'mfcc_duration': 2.5,
    'mfcc_sr': 44100,
    'mfcc_offset': 0.5,
    'n_mfcc': 13
}

def load_CREMA_data(in_path, info=False):
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
        path.append(CREMA + i)
    
    CREMA_df = pd.DataFrame(emotion, columns=['labels'])
    CREMA_df['source'] = 'CREMA'
    CREMA_df = pd.concat([CREMA_df, pd.DataFrame(path, columns=['path'])], axis=1)

    if info:
        print(CREMA_df.labels.value_counts())
        fname = CREMA + '1012_IEO_HAP_HI.wav'
        data, sampling_rate = librosa.load(fname)
        plt.figure(figsize=(15, 5))
        librosa.display.waveshow(data, sr=sampling_rate)
        plt.show()
    
    return CREMA_df

def load_SAVEE_data(in_path, info=False):
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

def load_RAVDESS_data(in_path, info=False):
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

def load_TESS_data(in_path, info=False):
    dir_list = os.listdir(in_path)
    dir_list.sort()
    path = []
    emotion = []

    for i in dir_list:
        fname = os.listdir(in_path + i)
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
            path.append(TESS + i + "/" + f)

    TESS_df = pd.DataFrame(emotion, columns = ['labels'])
    TESS_df['source'] = 'TESS'
    TESS_df = pd.concat([TESS_df, pd.DataFrame(path, columns = ['path'])], axis=1)
    return TESS_df

def extract_features(in_path, out_path, info=False):
    ref = pd.read_csv(in_path)
    ref.head()

    if info:
        print(ref)
    
    df = pd.DataFrame(columns=['feature'])
    counter = 0
    for index, path in tqdm(enumerate(ref.path), total=len(ref.path)):
        try:
            X, sample_rate = librosa.load(path, res_type='kaiser_fast', duration=config['mfcc_duration'], sr=config['mfcc_sr'], offset=config['mfcc_offset'])
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=config['n_mfcc']), axis=0)
            df.loc[counter] = [mfccs]
            counter += 1
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    df = pd.concat([ref, pd.DataFrame(df['feature'].values.tolist())], axis=1)
    df = df.fillna(0)
    df.to_csv(out_path, index=False)

class AudioModel(nn.Module):
    def __init__(self, num_classes):
        super(AudioModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 256, kernel_size=8, padding='same')
        self.conv2 = nn.Conv1d(256, 256, kernel_size=8, padding='same')
        self.bn1 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 128, kernel_size=8, padding='same')
        self.conv4 = nn.Conv1d(128, 128, kernel_size=8, padding='same')
        self.conv5 = nn.Conv1d(128, 128, kernel_size=8, padding='same')
        self.bn2 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 64, kernel_size=8, padding='same')
        self.conv7 = nn.Conv1d(64, 64, kernel_size=8, padding='same')
        self.fc = nn.Linear(192, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.maxpool = nn.MaxPool1d(8)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(self.relu(self.conv2(x)))
        x = self.dropout(self.maxpool(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.bn2(x)
        x = self.dropout(self.maxpool(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
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
        self.num_classes = len(np.unique(self.y))  # Dynamically get the number of classes
        self.y = torch.nn.functional.one_hot(torch.tensor(self.y), num_classes=self.num_classes).float()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]

def move_to_device(dataloader, device):
    data_on_device = []
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.unsqueeze(1)  # Add channel dimension
        data_on_device.append((inputs, labels))
    return data_on_device

def train_model(model, criterion, optimizer, train_data, num_epochs):
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

def validate_model(model, val_data, dataset):
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
    #print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(dataset.num_classes)]))

    return accuracy

def plot_training_progress(train_losses, train_accuracies):
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

def save_model(model, model_path, label_encoder, encoder_path):
    torch.save(model.state_dict(), model_path)
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoder, f)

def download_file_from_google_drive(url, output):
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)
    else:
        print(f"File {output} already exists. Skipping download.")

def load_model(model_path, encoder_path, num_classes, device):
    model = AudioModel(num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    with open(encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

def predict_emotion(audio_path, model, label_encoder):
    try:
        model.eval()
        X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=config['mfcc_duration'], sr=config['mfcc_sr'], offset=config['mfcc_offset'])
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=config['n_mfcc']), axis=0)
        newdf = pd.DataFrame(data=mfccs).T
        newdf = torch.tensor(newdf.values, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch and channel dimensions

        with torch.no_grad():
            newpred = model.forward(newdf)
            final = newpred.argmax(axis=1).cpu().numpy()
            final = final.astype(int).flatten()
            final = label_encoder.inverse_transform(final)
    except Exception as ex:
        return ['none_Unknown']
    
    return final

def aggregate_emotions(emotions):
    if not emotions:
        return 'Unknown'
    emotion_counts = {}
    for emotion in emotions:
        if emotion in emotion_counts:
            emotion_counts[emotion] += 1
        else:
            emotion_counts[emotion] = 1
    aggregated_emotion = max(emotion_counts, key=emotion_counts.get)
    return aggregated_emotion

def main():
    """cmdf, svdf, rvtf, tstf = load_CREMA_data(CREMA), load_SAVEE_data(SAVEE), load_RAVDESS_data(RAVDESS), load_TESS_data(TESS)
    df = pd.concat([cmdf, svdf, rvtf, tstf], axis=0)
    df.to_csv("data_path.csv", index=False)
    extract_features('data_path.csv', 'train_path.csv')"""

    dataset = AudioDataset('train_path.csv')
    train_size = int((config['train_size']) * len(dataset))
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_dataloader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    train_data_on_device = move_to_device(train_dataloader, device)
    val_data_on_device = move_to_device(val_dataloader, device)

    model = AudioModel(dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    model, train_losses, train_accuracies = train_model(model, criterion, optimizer, train_data_on_device, config['num_epochs'])

    plot_training_progress(train_losses, train_accuracies)

    validate_model(model, val_data_on_device, dataset)

    save_model(model, 'emotion_recog_model.pth', dataset.lb, 'label_encoder.pkl')

    """dataset = AudioDataset('train_path.csv')

    loaded_model, loaded_label_encoder = load_model('emotion_recog_model.pth', 'label_encoder.pkl', dataset.num_classes, device)

    sample_audio_path = 'audio/acc.wav'
    word_emotions = predict_emotion(sample_audio_path, loaded_model, loaded_label_encoder)
    sentence_emotion = aggregate_emotions(word_emotions)
    print(f'Predicted emotions for words: {word_emotions}')
    print(f'Aggregated emotion for the sentence: {sentence_emotion}')"""

def get_audio_emotion(audio_path):
    dataset = 14
    model_url = 'https://drive.google.com/uc?id=1KO-rIikXzCUkIkUWszuSTjotLuM2xpD2'
    encoder_url = 'https://drive.google.com/uc?id=1x4KYxSaXu-XbUi24aymXWANsRyNlVybr'
    model_path = '/tmp/emotion_recog_model.pth'
    encoder_path = '/tmp/label_encoder.pkl'
    
    download_file_from_google_drive(model_url, model_path)
    download_file_from_google_drive(encoder_url, encoder_path)

    loaded_model, loaded_label_encoder = load_model(model_path, encoder_path, dataset, device)
    
    audio = AudioSegment.from_file(audio_path)
    sentence_times = transcibe_audio(audio_path, 'sentence_verbose_json')
    
    emotions = []
    for sentence in sentence_times:
        start_ms = sentence['start'] * 1000
        end_ms = sentence['end'] * 1000
        audio_segment = audio[start_ms:end_ms]
        
        segment_path = "/tmp/temp_segment.wav"
        audio_segment.export(segment_path, format="wav")
        
        emotion = predict_emotion(segment_path, loaded_model, loaded_label_encoder)
        emotions.append({
            'text': sentence['text'],
            'start': sentence['start'],
            'end': sentence['end'],
            'emotion': emotion
        })
    
    return emotions

"""model_url = 'https://drive.google.com/uc?id=1KO-rIikXzCUkIkUWszuSTjotLuM2xpD2'
encoder_url = 'https://drive.google.com/uc?id=1x4KYxSaXu-XbUi24aymXWANsRyNlVybr'
model_path = '/tmp/emotion_recog_model.pth'
encoder_path = '/tmp/label_encoder.pkl'

download_file_from_google_drive(model_url, model_path)
download_file_from_google_drive(encoder_url, encoder_path)

loaded_model, loaded_label_encoder = load_model(model_path, encoder_path, 14, device)
print(predict_emotion('Ridiculous Name.wav', loaded_model, loaded_label_encoder))
print(get_audio_emotion('1AC vs nobody.wav'))"""

if __name__ == '__main__':
    main()