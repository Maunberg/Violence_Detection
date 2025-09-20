import librosa
import numpy as np
import torch
import torchaudio
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pickle
import os


class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, n_mfcc=13, n_fft=2048, hop_length=512):
        """
        Инициализация экстрактора аудио фичей
        
        Args:
            sample_rate: Частота дискретизации
            n_mfcc: Количество MFCC коэффициентов
            n_fft: Размер окна для FFT
            hop_length: Шаг между окнами
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        
    def extract_mfcc(self, audio_path):
        """Извлечение MFCC фичей"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, 
                                      n_fft=self.n_fft, hop_length=self.hop_length)
            return mfcc.T  # Транспонируем для временной оси
        except Exception as e:
            print(f"Ошибка при извлечении MFCC из {audio_path}: {e}")
            return None
    
    def extract_spectral_features(self, audio_path):
        """Извлечение спектральных фичей"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Спектральный центроид
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, 
                                                                 hop_length=self.hop_length)
            
            # Спектральная полоса пропускания
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, 
                                                                  hop_length=self.hop_length)
            
            # Спектральная контрастность
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, 
                                                                hop_length=self.hop_length)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y, hop_length=self.hop_length)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
            
            # Объединяем все фичи
            features = np.vstack([
                spectral_centroids,
                spectral_bandwidth,
                spectral_contrast,
                zcr,
                chroma,
                tonnetz
            ])
            
            return features.T
        except Exception as e:
            print(f"Ошибка при извлечении спектральных фичей из {audio_path}: {e}")
            return None
    
    def extract_rhythm_features(self, audio_path):
        """Извлечение ритмических фичей"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Tempo
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
            
            # Rhythm patterns
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=self.hop_length)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=self.hop_length)
            
            # Rhythm strength
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
            
            return {
                'tempo': tempo,
                'beats': beats,
                'onset_times': onset_times,
                'onset_strength': onset_strength
            }
        except Exception as e:
            print(f"Ошибка при извлечении ритмических фичей из {audio_path}: {e}")
            return None
    
    def extract_all_features(self, audio_path):
        """Извлечение всех аудио фичей"""
        features = {}
        
        # MFCC
        mfcc = self.extract_mfcc(audio_path)
        if mfcc is not None:
            features['mfcc'] = mfcc
            features['mfcc_mean'] = np.mean(mfcc, axis=0)
            features['mfcc_std'] = np.std(mfcc, axis=0)
        
        # Спектральные фичи
        spectral = self.extract_spectral_features(audio_path)
        if spectral is not None:
            features['spectral'] = spectral
            features['spectral_mean'] = np.mean(spectral, axis=0)
            features['spectral_std'] = np.std(spectral, axis=0)
        
        # Ритмические фичи
        rhythm = self.extract_rhythm_features(audio_path)
        if rhythm is not None:
            features['rhythm'] = rhythm
        
        return features


def extract_audio_from_video(video_path, audio_path):
    """Извлечение аудио из видео файла"""
    try:
        import subprocess
        cmd = [
            'ffmpeg', '-i', video_path, '-vn', '-acodec', 'pcm_s16le', 
            '-ar', '22050', '-ac', '1', '-y', audio_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"Ошибка при извлечении аудио из {video_path}: {e}")
        return False


def process_video_audio_features(df_path, output_dir="audio_features"):
    """
    Обработка всех видео файлов и извлечение аудио фичей
    
    Args:
        df_path: Путь к CSV файлу с данными
        output_dir: Директория для сохранения фичей
    """
    # Создаем директорию для аудио фичей
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/temp_audio", exist_ok=True)
    
    # Загружаем данные
    df = pd.read_csv(df_path)
    
    # Инициализируем экстрактор
    extractor = AudioFeatureExtractor()
    
    audio_features_dict = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Извлечение аудио фичей"):
        video_path = row['way']
        tag = row['teg']
        part = row['part']
        
        # Создаем уникальный ключ
        video_name = Path(video_path).stem
        key = f"{tag}_{part}_{video_name}"
        
        # Временный путь для аудио
        temp_audio_path = f"{output_dir}/temp_audio/{video_name}.wav"
        
        # Извлекаем аудио из видео
        if extract_audio_from_video(video_path, temp_audio_path):
            # Извлекаем фичи
            features = extractor.extract_all_features(temp_audio_path)
            if features:
                audio_features_dict[key] = features
            
            # Удаляем временный аудио файл
            if os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)
    
    # Сохраняем фичи
    with open(f"{output_dir}/audio_features.pkl", "wb") as f:
        pickle.dump(audio_features_dict, f)
    
    print(f"Аудио фичи сохранены в {output_dir}/audio_features.pkl")
    print(f"Обработано {len(audio_features_dict)} видео файлов")
    
    return audio_features_dict


if __name__ == "__main__":
    # Обрабатываем аудио фичи
    audio_features = process_video_audio_features("df.csv")
