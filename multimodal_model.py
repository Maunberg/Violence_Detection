import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from pathlib import Path


class MultimodalDataset(Dataset):
    """Датасет для мультимодальных данных"""
    
    def __init__(self, audio_features, visual_features, labels, scaler_audio=None, scaler_visual=None, fit_scalers=True):
        """
        Args:
            audio_features: Словарь с аудио фичами
            visual_features: Словарь с визуальными фичами
            labels: Словарь с метками (0 - non_violent, 1 - violent)
            scaler_audio: Предобученный скейлер для аудио фичей
            scaler_visual: Предобученный скейлер для визуальных фичей
            fit_scalers: Обучать ли скейлеры на данных
        """
        self.audio_features = audio_features
        self.visual_features = visual_features
        self.labels = labels
        
        # Получаем общие ключи
        self.keys = list(set(audio_features.keys()) & set(visual_features.keys()) & set(labels.keys()))
        
        # Подготавливаем фичи
        self.audio_data, self.visual_data, self.label_data = self._prepare_features()
        
        # Создаем скейлеры
        if fit_scalers:
            self.scaler_audio = StandardScaler()
            self.scaler_visual = StandardScaler()
            self.audio_data = self.scaler_audio.fit_transform(self.audio_data)
            self.visual_data = self.scaler_visual.fit_transform(self.visual_data)
        else:
            self.scaler_audio = scaler_audio
            self.scaler_visual = scaler_visual
            if scaler_audio is not None and scaler_visual is not None:
                self.audio_data = self.scaler_audio.transform(self.audio_data)
                self.visual_data = self.scaler_visual.transform(self.visual_data)
    
    def _prepare_features(self):
        """Подготовка фичей для обучения"""
        audio_data = []
        visual_data = []
        label_data = []
        
        for key in self.keys:
            # Аудио фичи
            audio_feat = self.audio_features[key]
            audio_vector = self._extract_audio_vector(audio_feat)
            if audio_vector is not None:
                audio_data.append(audio_vector)
            else:
                continue
            
            # Визуальные фичи
            visual_feat = self.visual_features[key]
            visual_vector = self._extract_visual_vector(visual_feat)
            if visual_vector is not None:
                visual_data.append(visual_vector)
            else:
                continue
            
            # Метка
            label_data.append(self.labels[key])
        
        return np.array(audio_data), np.array(visual_data), np.array(label_data)
    
    def _extract_audio_vector(self, audio_feat):
        """Извлечение вектора из аудио фичей"""
        try:
            vector = []
            
            # MFCC статистики (фиксированный размер - 13 коэффициентов)
            if 'mfcc_mean' in audio_feat:
                mfcc_mean = audio_feat['mfcc_mean']
                if isinstance(mfcc_mean, np.ndarray):
                    # Берем первые 13 элементов или дополняем нулями
                    mfcc_mean = mfcc_mean[:13] if len(mfcc_mean) >= 13 else np.pad(mfcc_mean, (0, 13 - len(mfcc_mean)), 'constant')
                vector.extend(mfcc_mean)
            else:
                vector.extend([0] * 13)
                
            if 'mfcc_std' in audio_feat:
                mfcc_std = audio_feat['mfcc_std']
                if isinstance(mfcc_std, np.ndarray):
                    mfcc_std = mfcc_std[:13] if len(mfcc_std) >= 13 else np.pad(mfcc_std, (0, 13 - len(mfcc_std)), 'constant')
                vector.extend(mfcc_std)
            else:
                vector.extend([0] * 13)
            
            # Спектральные статистики (фиксированный размер - 20 элементов)
            if 'spectral_mean' in audio_feat:
                spectral_mean = audio_feat['spectral_mean']
                if isinstance(spectral_mean, np.ndarray):
                    spectral_mean = spectral_mean[:20] if len(spectral_mean) >= 20 else np.pad(spectral_mean, (0, 20 - len(spectral_mean)), 'constant')
                vector.extend(spectral_mean)
            else:
                vector.extend([0] * 20)
                
            if 'spectral_std' in audio_feat:
                spectral_std = audio_feat['spectral_std']
                if isinstance(spectral_std, np.ndarray):
                    spectral_std = spectral_std[:20] if len(spectral_std) >= 20 else np.pad(spectral_std, (0, 20 - len(spectral_std)), 'constant')
                vector.extend(spectral_std)
            else:
                vector.extend([0] * 20)
            
            # Ритмические фичи (фиксированный размер - 4 элемента)
            if 'rhythm' in audio_feat:
                rhythm = audio_feat['rhythm']
                vector.append(float(rhythm.get('tempo', 0)))
                if 'onset_strength' in rhythm and rhythm['onset_strength'] is not None:
                    onset_strength = rhythm['onset_strength']
                    if isinstance(onset_strength, np.ndarray) and len(onset_strength) > 0:
                        vector.extend([
                            float(np.mean(onset_strength)),
                            float(np.std(onset_strength)),
                            float(np.max(onset_strength))
                        ])
                    else:
                        vector.extend([0.0, 0.0, 0.0])
                else:
                    vector.extend([0.0, 0.0, 0.0])
            else:
                vector.extend([0.0, 0.0, 0.0, 0.0])
            
            return np.array(vector, dtype=np.float32)
        except Exception as e:
            print(f"Ошибка при извлечении аудио вектора: {e}")
            return None
    
    def _extract_visual_vector(self, visual_feat):
        """Извлечение вектора из визуальных фичей"""
        try:
            vector = []
            
            # CNN фичи (фиксированный размер - 2048 для ResNet50)
            if 'cnn_mean' in visual_feat:
                cnn_mean = visual_feat['cnn_mean']
                if isinstance(cnn_mean, np.ndarray):
                    cnn_mean = cnn_mean[:2048] if len(cnn_mean) >= 2048 else np.pad(cnn_mean, (0, 2048 - len(cnn_mean)), 'constant')
                vector.extend(cnn_mean)
            else:
                vector.extend([0] * 2048)
                
            if 'cnn_std' in visual_feat:
                cnn_std = visual_feat['cnn_std']
                if isinstance(cnn_std, np.ndarray):
                    cnn_std = cnn_std[:2048] if len(cnn_std) >= 2048 else np.pad(cnn_std, (0, 2048 - len(cnn_std)), 'constant')
                vector.extend(cnn_std)
            else:
                vector.extend([0] * 2048)
                
            if 'cnn_max' in visual_feat:
                cnn_max = visual_feat['cnn_max']
                if isinstance(cnn_max, np.ndarray):
                    cnn_max = cnn_max[:2048] if len(cnn_max) >= 2048 else np.pad(cnn_max, (0, 2048 - len(cnn_max)), 'constant')
                vector.extend(cnn_max)
            else:
                vector.extend([0] * 2048)
            
            # Цветовые фичи (фиксированный размер - 3 для RGB)
            if 'color_mean' in visual_feat:
                color_mean = visual_feat['color_mean']
                if isinstance(color_mean, np.ndarray):
                    color_mean = color_mean[:3] if len(color_mean) >= 3 else np.pad(color_mean, (0, 3 - len(color_mean)), 'constant')
                vector.extend(color_mean)
            else:
                vector.extend([0] * 3)
                
            if 'color_std' in visual_feat:
                color_std = visual_feat['color_std']
                if isinstance(color_std, np.ndarray):
                    color_std = color_std[:3] if len(color_std) >= 3 else np.pad(color_std, (0, 3 - len(color_std)), 'constant')
                vector.extend(color_std)
            else:
                vector.extend([0] * 3)
            
            # Текстурные фичи (фиксированный размер - 2 элемента)
            if 'texture_gradient_mean' in visual_feat:
                vector.append(float(visual_feat['texture_gradient_mean']))
            else:
                vector.append(0.0)
            
            if 'texture_gradient_std' in visual_feat:
                vector.append(float(visual_feat['texture_gradient_std']))
            else:
                vector.append(0.0)
            
            # Фичи движения (фиксированный размер - 3 элемента)
            if 'motion' in visual_feat:
                motion = visual_feat['motion']
                vector.extend([
                    float(motion.get('motion_mean', 0)),
                    float(motion.get('motion_std', 0)),
                    float(motion.get('motion_max', 0))
                ])
            else:
                vector.extend([0.0, 0.0, 0.0])
            
            return np.array(vector, dtype=np.float32)
        except Exception as e:
            print(f"Ошибка при извлечении визуального вектора: {e}")
            return None
    
    def __len__(self):
        return len(self.audio_data)
    
    def __getitem__(self, idx):
        return {
            'audio': torch.FloatTensor(self.audio_data[idx]),
            'visual': torch.FloatTensor(self.visual_data[idx]),
            'label': torch.LongTensor([self.label_data[idx]]).squeeze()
        }


class AudioEncoder(nn.Module):
    """Энкодер для аудио фичей"""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128]):
        super(AudioEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Заменяем BatchNorm на LayerNorm
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        return self.encoder(x)


class VisualEncoder(nn.Module):
    """Энкодер для визуальных фичей"""
    
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256]):
        super(VisualEncoder, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Заменяем BatchNorm на LayerNorm
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
    
    def forward(self, x):
        return self.encoder(x)


class AttentionFusion(nn.Module):
    """Модуль внимания для слияния модальностей"""
    
    def __init__(self, audio_dim, visual_dim, hidden_dim=128):
        super(AttentionFusion, self).__init__()
        
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, audio_feat, visual_feat):
        # Проецируем фичи в общее пространство
        audio_proj = self.audio_proj(audio_feat).unsqueeze(1)  # [batch, 1, hidden_dim]
        visual_proj = self.visual_proj(visual_feat).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Объединяем фичи
        combined = torch.cat([audio_proj, visual_proj], dim=1)  # [batch, 2, hidden_dim]
        
        # Применяем self-attention
        attended, _ = self.attention(combined, combined, combined)
        
        # Нормализация
        attended = self.norm(attended)
        
        # Глобальное усреднение
        fused = torch.mean(attended, dim=1)  # [batch, hidden_dim]
        
        return fused


class MultimodalViolenceClassifier(nn.Module):
    """Мультимодальная модель для классификации насилия"""
    
    def __init__(self, audio_input_dim, visual_input_dim, num_classes=2, 
                 audio_hidden_dims=[512, 256, 128], visual_hidden_dims=[1024, 512, 256],
                 fusion_hidden_dim=128, classifier_hidden_dims=[256, 128]):
        super(MultimodalViolenceClassifier, self).__init__()
        
        # Энкодеры для каждой модальности
        self.audio_encoder = AudioEncoder(audio_input_dim, audio_hidden_dims)
        self.visual_encoder = VisualEncoder(visual_input_dim, visual_hidden_dims)
        
        # Модуль слияния с вниманием
        self.fusion = AttentionFusion(
            self.audio_encoder.output_dim, 
            self.visual_encoder.output_dim, 
            fusion_hidden_dim
        )
        
        # Классификатор
        classifier_layers = []
        prev_dim = fusion_hidden_dim
        
        for hidden_dim in classifier_hidden_dims:
            classifier_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),  # Заменяем BatchNorm на LayerNorm
                nn.ReLU(),
                nn.Dropout(0.5)
            ])
            prev_dim = hidden_dim
        
        classifier_layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*classifier_layers)
    
    def forward(self, audio_input, visual_input):
        # Кодируем каждую модальность
        audio_features = self.audio_encoder(audio_input)
        visual_features = self.visual_encoder(visual_input)
        
        # Сливаем модальности с вниманием
        fused_features = self.fusion(audio_features, visual_features)
        
        # Классификация
        output = self.classifier(fused_features)
        
        return output
    
    def get_embeddings(self, audio_input, visual_input):
        """Получение эмбеддингов для анализа"""
        with torch.no_grad():
            audio_features = self.audio_encoder(audio_input)
            visual_features = self.visual_encoder(visual_input)
            fused_features = self.fusion(audio_features, visual_features)
            return fused_features


def create_data_loaders(audio_features_path, visual_features_path, df_path, 
                       test_size=0.2, val_size=0.1, batch_size=32, random_state=42):
    """Создание загрузчиков данных"""
    
    # Загружаем фичи
    with open(audio_features_path, 'rb') as f:
        audio_features = pickle.load(f)
    
    with open(visual_features_path, 'rb') as f:
        visual_features = pickle.load(f)
    
    # Загружаем метаданные
    df = pd.read_csv(df_path)
    
    # Создаем метки
    labels = {}
    for _, row in df.iterrows():
        video_name = Path(row['way']).stem
        key = f"{row['teg']}_{row['part']}_{video_name}"
        labels[key] = 1 if row['teg'] == 'violent' else 0
    
    # Получаем общие ключи
    common_keys = set(audio_features.keys()) & set(visual_features.keys()) & set(labels.keys())
    
    # Фильтруем данные
    audio_features_filtered = {k: audio_features[k] for k in common_keys}
    visual_features_filtered = {k: visual_features[k] for k in common_keys}
    labels_filtered = {k: labels[k] for k in common_keys}
    
    print(f"Общее количество образцов: {len(common_keys)}")
    print(f"Violent: {sum(labels_filtered.values())}")
    print(f"Non-violent: {len(labels_filtered) - sum(labels_filtered.values())}")
    
    # Разделяем на train/val/test
    keys_list = list(common_keys)
    
    # Для малых датасетов используем простое разделение
    if len(keys_list) < 10:
        # Простое разделение для малых датасетов
        n_train = max(1, int(len(keys_list) * 0.6))
        n_val = max(1, int(len(keys_list) * 0.2))
        
        train_keys = keys_list[:n_train]
        val_keys = keys_list[n_train:n_train + n_val]
        test_keys = keys_list[n_train + n_val:]
        
        print(f"Простое разделение для малого датасета:")
        print(f"  - Train: {len(train_keys)}")
        print(f"  - Val: {len(val_keys)}")
        print(f"  - Test: {len(test_keys)}")
    else:
        # Стандартное разделение для больших датасетов
        train_keys, temp_keys = train_test_split(keys_list, test_size=test_size + val_size, 
                                               random_state=random_state, stratify=[labels_filtered[k] for k in keys_list])
        val_keys, test_keys = train_test_split(temp_keys, test_size=test_size/(test_size + val_size), 
                                             random_state=random_state, stratify=[labels_filtered[k] for k in temp_keys])
    
    # Создаем датасеты
    train_audio = {k: audio_features_filtered[k] for k in train_keys}
    train_visual = {k: visual_features_filtered[k] for k in train_keys}
    train_labels = {k: labels_filtered[k] for k in train_keys}
    
    val_audio = {k: audio_features_filtered[k] for k in val_keys}
    val_visual = {k: visual_features_filtered[k] for k in val_keys}
    val_labels = {k: labels_filtered[k] for k in val_keys}
    
    test_audio = {k: audio_features_filtered[k] for k in test_keys}
    test_visual = {k: visual_features_filtered[k] for k in test_keys}
    test_labels = {k: labels_filtered[k] for k in test_keys}
    
    # Создаем датасеты
    train_dataset = MultimodalDataset(train_audio, train_visual, train_labels, fit_scalers=True)
    val_dataset = MultimodalDataset(val_audio, val_visual, val_labels, 
                                  scaler_audio=train_dataset.scaler_audio, 
                                  scaler_visual=train_dataset.scaler_visual, 
                                  fit_scalers=False)
    test_dataset = MultimodalDataset(test_audio, test_visual, test_labels,
                                   scaler_audio=train_dataset.scaler_audio,
                                   scaler_visual=train_dataset.scaler_visual,
                                   fit_scalers=False)
    
    # Создаем загрузчики
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.scaler_audio, train_dataset.scaler_visual


if __name__ == "__main__":
    # Пример использования
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Используется устройство: {device}")
    
    # Создаем загрузчики данных
    train_loader, val_loader, test_loader, scaler_audio, scaler_visual = create_data_loaders(
        "audio_features/audio_features.pkl",
        "visual_features/visual_features.pkl", 
        "df.csv"
    )
    
    # Получаем размеры входных данных
    sample_batch = next(iter(train_loader))
    audio_dim = sample_batch['audio'].shape[1]
    visual_dim = sample_batch['visual'].shape[1]
    
    print(f"Размер аудио фичей: {audio_dim}")
    print(f"Размер визуальных фичей: {visual_dim}")
    
    # Создаем модель с правильными размерами
    model = MultimodalViolenceClassifier(audio_dim, visual_dim).to(device)
    print(f"Модель создана. Параметров: {sum(p.numel() for p in model.parameters())}")
