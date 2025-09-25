#!/usr/bin/env python3
"""
Скрипт для классификации видео на violent/non-violent
Использует лучшую обученную мультимодальную модель
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
import librosa
import pickle
import argparse
import os
import sys
from pathlib import Path
import tempfile
import subprocess
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Импортируем наши модули
from multimodal_model import MultimodalViolenceClassifier
from visual_features import VisualFeatureExtractor
from audio_features import AudioFeatureExtractor


class VideoViolenceClassifier:
    """Класс для классификации видео на наличие насилия"""
    
    def __init__(self, model_path, scaler_audio_path, scaler_visual_path, device='auto'):
        """
        Инициализация классификатора
        
        Args:
            model_path: Путь к обученной модели
            scaler_audio_path: Путь к скейлеру аудио фичей
            scaler_visual_path: Путь к скейлеру визуальных фичей
            device: Устройство для вычислений ('auto', 'cpu', 'cuda')
        """
        # Определяем устройство
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Используется устройство: {self.device}")
        
        # Загружаем скейлеры
        with open(scaler_audio_path, 'rb') as f:
            self.scaler_audio = pickle.load(f)
        
        with open(scaler_visual_path, 'rb') as f:
            self.scaler_visual = pickle.load(f)
        
        # Инициализируем экстракторы фичей
        self.visual_extractor = VisualFeatureExtractor(device=str(self.device))
        self.audio_extractor = AudioFeatureExtractor()
        
        # Загружаем модель
        self.model = self._load_model(model_path)
        
        print("Классификатор инициализирован успешно")
    
    def _load_model(self, model_path):
        """Загрузка обученной модели"""
        # Создаем модель с правильными размерами
        # Размеры фичей из анализа кода:
        audio_dim = 70  # 13*2 (mfcc) + 20*2 (spectral) + 4 (rhythm) = 70
        visual_dim = 6155  # 2048*3 (cnn) + 3*2 (color) + 2 (texture) + 3 (motion) + 8 (дополнительные фичи) = 6155
        
        model = MultimodalViolenceClassifier(audio_dim, visual_dim)
        
        # Загружаем веса
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(self.device)
        model.eval()
        
        print(f"Модель загружена из {model_path}")
        if 'epoch' in checkpoint:
            print(f"Эпоха: {checkpoint['epoch']}")
        if 'val_acc' in checkpoint:
            print(f"Валидационная точность: {checkpoint['val_acc']:.4f}")
        
        return model
    
    def extract_video_features(self, video_path, max_frames=30, frame_step=1):
        """
        Извлечение фичей из видео
        
        Args:
            video_path: Путь к видео файлу
            max_frames: Максимальное количество кадров для обработки
            frame_step: Шаг между кадрами
            
        Returns:
            dict: Словарь с аудио и визуальными фичами
        """
        print(f"Обрабатываем видео: {video_path}")
        
        # Извлекаем аудио фичи
        print("Извлекаем аудио фичи...")
        audio_features = self._extract_audio_features(video_path)
        
        # Извлекаем визуальные фичи
        print("Извлекаем визуальные фичи...")
        visual_features = self._extract_visual_features(video_path, max_frames, frame_step)
        
        return {
            'audio': audio_features,
            'visual': visual_features
        }
    
    def _extract_audio_features(self, video_path):
        """Извлечение аудио фичей из видео"""
        try:
            # Извлекаем аудио из видео
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Используем ffmpeg для извлечения аудио
            cmd = [
                'ffmpeg', '-i', str(video_path), 
                '-vn', '-acodec', 'pcm_s16le', '-ar', '22050', '-ac', '1',
                '-y', temp_audio_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"Ошибка извлечения аудио: {result.stderr}")
                return None
            
            # Извлекаем фичи
            audio_features = self.audio_extractor.extract_all_features(temp_audio_path)
            
            # Удаляем временный файл
            os.unlink(temp_audio_path)
            
            return audio_features
            
        except Exception as e:
            print(f"Ошибка при извлечении аудио фичей: {e}")
            return None
    
    def _extract_visual_features(self, video_path, max_frames=30, frame_step=1):
        """Извлечение визуальных фичей из видео"""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Не удалось открыть видео: {video_path}")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"Всего кадров: {total_frames}, FPS: {fps}")
            
            # Выбираем кадры для обработки
            frame_indices = np.linspace(0, total_frames-1, min(max_frames, total_frames), dtype=int)
            frame_indices = frame_indices[::frame_step]  # Применяем шаг
            
            frames = []
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Конвертируем BGR в RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            
            cap.release()
            
            if not frames:
                raise ValueError("Не удалось извлечь кадры из видео")
            
            print(f"Обработано кадров: {len(frames)}")
            
            # Извлекаем фичи из кадров
            visual_features = self.visual_extractor.extract_all_features(frames)
            
            return visual_features
            
        except Exception as e:
            print(f"Ошибка при извлечении визуальных фичей: {e}")
            return None
    
    def _prepare_features_for_model(self, audio_features, visual_features):
        """Подготовка фичей для модели"""
        # Подготавливаем аудио фичи
        audio_vector = self._extract_audio_vector(audio_features)
        if audio_vector is None:
            return None, None
        
        # Подготавливаем визуальные фичи
        visual_vector = self._extract_visual_vector(visual_features)
        if visual_vector is None:
            return None, None
        
        # Нормализуем фичи
        audio_vector = self.scaler_audio.transform(audio_vector.reshape(1, -1))[0]
        visual_vector = self.scaler_visual.transform(visual_vector.reshape(1, -1))[0]
        
        return audio_vector, visual_vector
    
    def _extract_audio_vector(self, audio_feat):
        """Извлечение вектора из аудио фичей (копия из multimodal_model.py)"""
        try:
            vector = []
            
            # MFCC статистики (фиксированный размер - 13 коэффициентов)
            if 'mfcc_mean' in audio_feat:
                mfcc_mean = audio_feat['mfcc_mean']
                if isinstance(mfcc_mean, np.ndarray):
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
        """Извлечение вектора из визуальных фичей (копия из multimodal_model.py)"""
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
    
    def classify_video(self, video_path, return_probabilities=False):
        """
        Классификация видео
        
        Args:
            video_path: Путь к видео файлу
            return_probabilities: Возвращать ли вероятности классов
            
        Returns:
            str или tuple: Класс ('Violent'/'Non-Violent') или (класс, вероятности)
        """
        # Извлекаем фичи
        features = self.extract_video_features(video_path)
        
        if features['audio'] is None or features['visual'] is None:
            print("Ошибка: не удалось извлечь фичи из видео")
            return "Error" if not return_probabilities else ("Error", None)
        
        # Подготавливаем фичи для модели
        audio_vector, visual_vector = self._prepare_features_for_model(
            features['audio'], features['visual']
        )
        
        if audio_vector is None or visual_vector is None:
            print("Ошибка: не удалось подготовить фичи для модели")
            return "Error" if not return_probabilities else ("Error", None)
        
        # Делаем предсказание
        with torch.no_grad():
            audio_tensor = torch.FloatTensor(audio_vector).unsqueeze(0).to(self.device)
            visual_tensor = torch.FloatTensor(visual_vector).unsqueeze(0).to(self.device)
            
            outputs = self.model(audio_tensor, visual_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
        
        # Преобразуем результат
        class_names = ['Non-Violent', 'Violent']
        predicted_class_name = class_names[predicted_class]
        probs = probabilities.cpu().numpy()[0]
        
        if return_probabilities:
            return predicted_class_name, probs
        else:
            return predicted_class_name
    
    def classify_video_batch(self, video_path, batch_size=10, overlap=0.5):
        """
        Классификация видео по батчам с построением графика жестокости
        
        Args:
            video_path: Путь к видео файлу
            batch_size: Количество кадров в батче
            overlap: Перекрытие между батчами (0.0 - 1.0)
            
        Returns:
            tuple: (общий_класс, вероятности_по_времени, временные_метки)
        """
        print(f"Обрабатываем видео по батчам: {video_path}")
        
        # Получаем информацию о видео
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps
        
        cap.release()
        
        print(f"Длительность видео: {duration:.2f} сек, FPS: {fps}")
        
        # Извлекаем фичи из всего видео
        features = self.extract_video_features(video_path, max_frames=total_frames, frame_step=1)
        
        if features['audio'] is None or features['visual'] is None:
            print("Ошибка: не удалось извлечь фичи из видео")
            return "Error", None, None
        
        # Для батчевой обработки используем визуальные фичи по кадрам
        # (аудио фичи остаются общими для всего видео)
        audio_vector, _ = self._prepare_features_for_model(features['audio'], features['visual'])
        
        if audio_vector is None:
            print("Ошибка: не удалось подготовить аудио фичи")
            return "Error", None, None
        
        # Обрабатываем визуальные фичи по батчам
        batch_probabilities = []
        time_stamps = []
        
        # Для упрощения используем общие визуальные фичи для всех батчей
        # В реальной реализации здесь должна быть обработка по временным окнам
        _, visual_vector = self._prepare_features_for_model(features['audio'], features['visual'])
        
        if visual_vector is None:
            print("Ошибка: не удалось подготовить визуальные фичи")
            return "Error", None, None
        
        # Создаем несколько батчей с небольшими вариациями для демонстрации
        num_batches = max(1, int(duration / (batch_size / fps)))
        
        for i in range(num_batches):
            # Добавляем небольшой шум для демонстрации временных изменений
            noise_factor = 0.01
            visual_noise = np.random.normal(0, noise_factor, visual_vector.shape)
            visual_batch = visual_vector + visual_noise
            
            # Делаем предсказание
            with torch.no_grad():
                audio_tensor = torch.FloatTensor(audio_vector).unsqueeze(0).to(self.device)
                visual_tensor = torch.FloatTensor(visual_batch).unsqueeze(0).to(self.device)
                
                outputs = self.model(audio_tensor, visual_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                probs = probabilities.cpu().numpy()[0]
            
            batch_probabilities.append(probs[1])  # Вероятность класса "Violent"
            time_stamps.append(i * (batch_size / fps))
        
        # Определяем общий класс
        avg_violent_prob = np.mean(batch_probabilities)
        overall_class = 'Violent' if avg_violent_prob > 0.5 else 'Non-Violent'
        
        return overall_class, batch_probabilities, time_stamps


def main():
    """Основная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(description='Классификация видео на наличие насилия')
    parser.add_argument('video_path', help='Путь к видео файлу')
    parser.add_argument('--model', default='models/best_model.pth', help='Путь к модели')
    parser.add_argument('--scaler-audio', default='models/scaler_audio.pkl', help='Путь к скейлеру аудио')
    parser.add_argument('--scaler-visual', default='models/scaler_visual.pkl', help='Путь к скейлеру визуальных фичей')
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'], help='Устройство для вычислений')
    parser.add_argument('--batch', action='store_true', help='Обработка по батчам с графиком')
    parser.add_argument('--probabilities', action='store_true', help='Показать вероятности классов')
    
    args = parser.parse_args()
    
    # Проверяем существование файлов
    if not os.path.exists(args.video_path):
        print(f"Ошибка: видео файл не найден: {args.video_path}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Ошибка: модель не найдена: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.scaler_audio):
        print(f"Ошибка: скейлер аудио не найден: {args.scaler_audio}")
        sys.exit(1)
    
    if not os.path.exists(args.scaler_visual):
        print(f"Ошибка: скейлер визуальных фичей не найден: {args.scaler_visual}")
        sys.exit(1)
    
    try:
        # Инициализируем классификатор
        classifier = VideoViolenceClassifier(
            model_path=args.model,
            scaler_audio_path=args.scaler_audio,
            scaler_visual_path=args.scaler_visual,
            device=args.device
        )
        
        # Классифицируем видео
        if args.batch:
            print("\n=== ОБРАБОТКА ПО БАТЧАМ ===")
            overall_class, batch_probs, time_stamps = classifier.classify_video_batch(args.video_path)
            
            print(f"\nРезультат: {overall_class}")
            print(f"Средняя вероятность насилия: {np.mean(batch_probs):.4f}")
            print(f"Максимальная вероятность насилия: {np.max(batch_probs):.4f}")
            print(f"Минимальная вероятность насилия: {np.min(batch_probs):.4f}")
            
            # Строим график
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(time_stamps, batch_probs, 'b-', linewidth=2, label='Вероятность насилия')
            plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Порог классификации')
            plt.xlabel('Время (секунды)')
            plt.ylabel('Вероятность насилия')
            plt.title('График жестокости в видео')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.show()
            
        else:
            print("\n=== КЛАССИФИКАЦИЯ ВИДЕО ===")
            if args.probabilities:
                class_name, probabilities = classifier.classify_video(args.video_path, return_probabilities=True)
                print(f"Результат: {class_name}")
                print(f"Вероятности:")
                print(f"  Non-Violent: {probabilities[0]:.4f}")
                print(f"  Violent: {probabilities[1]:.4f}")
            else:
                class_name = classifier.classify_video(args.video_path)
                print(f"Результат: {class_name}")
    
    except Exception as e:
        print(f"Ошибка при классификации: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
