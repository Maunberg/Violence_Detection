import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, mobilenet_v3_small
import torch.nn as nn
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pickle
import os
from PIL import Image
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time


class VisualFeatureExtractor:
    def __init__(self, device='cpu', model_type='mobilenet', batch_size=8):
        """
        Оптимизированный экстрактор визуальных фичей
        
        Args:
            device: Устройство для вычислений ('cpu' или 'cuda')
            model_type: Тип модели ('resnet50', 'mobilenet')
            batch_size: Размер батча для обработки
        """
        self.device = torch.device(device)
        self.model_type = model_type
        self.batch_size = batch_size
        
        # Загружаем предобученную модель
        if model_type == 'resnet50':
            self.model = resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        elif model_type == 'mobilenet':
            self.model = mobilenet_v3_small(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 576
        else:
            raise ValueError("Поддерживаются только 'resnet50' и 'mobilenet'")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Оптимизированные трансформации
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Кэш для избежания повторных вычислений
        self.feature_cache = {}
    
    def preprocess_frames_batch(self, frames):
        """Быстрая предобработка батча кадров"""
        batch_tensors = []
        
        for frame in frames:
            try:
                # Быстрая конвертация BGR -> RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Применяем трансформации
                tensor = self.transform(pil_image)
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Ошибка предобработки кадра: {e}")
                # Добавляем нулевой тензор в случае ошибки
                batch_tensors.append(torch.zeros(3, 224, 224))
        
        return torch.stack(batch_tensors)
    
    def extract_cnn_features_batch(self, frames):
        """Извлечение CNN фичей батчами для ускорения"""
        if not frames:
            return None
        
        all_features = []
        
        # Обрабатываем кадры батчами
        for i in range(0, len(frames), self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            batch_tensor = self.preprocess_frames_batch(batch_frames).to(self.device)
            
            with torch.no_grad():
                batch_features = self.model(batch_tensor)
                batch_features = batch_features.squeeze().cpu().numpy()
                
                # Если батч содержит один элемент, добавляем размерность
                if len(batch_features.shape) == 1:
                    batch_features = batch_features.reshape(1, -1)
                
                all_features.append(batch_features)
        
        if all_features:
            return np.vstack(all_features)
        return None
    
    def extract_simple_motion_features(self, frames):
        """Упрощенное извлечение фичей движения"""
        try:
            if len(frames) < 2:
                return None
            
            # Берем только несколько пар кадров
            step = max(1, len(frames) // 10)  # Максимум 10 пар
            motion_values = []
            
            for i in range(step, len(frames), step):
                prev_gray = cv2.cvtColor(frames[i-step], cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
                
                # Простое вычисление разности
                diff = cv2.absdiff(prev_gray, curr_gray)
                motion_magnitude = np.mean(diff)
                motion_values.append(motion_magnitude)
            
            if motion_values:
                return {
                    'motion_mean': np.mean(motion_values),
                    'motion_std': np.std(motion_values),
                    'motion_max': np.max(motion_values)
                }
            return None
        except Exception as e:
            print(f"Ошибка при извлечении фичей движения: {e}")
            return None
    
    def extract_fast_color_features(self, frames):
        """Быстрое извлечение цветовых фичей"""
        try:
            if not frames:
                return None
            
            # Берем только несколько кадров для анализа цвета
            sample_frames = frames[::max(1, len(frames) // 5)]  # Максимум 5 кадров
            
            color_stats = []
            
            for frame in sample_frames:
                # Быстрые статистики по цветам
                mean_color = np.mean(frame.reshape(-1, 3), axis=0)
                std_color = np.std(frame.reshape(-1, 3), axis=0)
                
                # Простая гистограмма по яркости
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                hist = cv2.calcHist([gray], [0], None, [32], [0, 256])  # Уменьшили bins
                hist = hist.flatten() / hist.sum()  # Нормализация
                
                color_stats.append({
                    'mean_color': mean_color,
                    'std_color': std_color,
                    'brightness_hist': hist
                })
            
            # Агрегируем статистики
            mean_colors = np.array([cs['mean_color'] for cs in color_stats])
            std_colors = np.array([cs['std_color'] for cs in color_stats])
            histograms = np.array([cs['brightness_hist'] for cs in color_stats])
            
            return {
                'color_mean': np.mean(mean_colors, axis=0),
                'color_std': np.mean(std_colors, axis=0),
                'brightness_hist_mean': np.mean(histograms, axis=0),
                'brightness_hist_std': np.std(histograms, axis=0)
            }
        except Exception as e:
            print(f"Ошибка при извлечении цветовых фичей: {e}")
            return None
    
    def extract_fast_texture_features(self, frames):
        """Быстрое извлечение текстурных фичей"""
        try:
            if not frames:
                return None
            
            # Берем только несколько кадров
            sample_frames = frames[::max(1, len(frames) // 3)]  # Максимум 3 кадра
            
            texture_stats = []
            
            for frame in sample_frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Быстрые градиенты
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                
                # Простые статистики
                texture_stats.append({
                    'gradient_mean': np.mean(gradient_magnitude),
                    'gradient_std': np.std(gradient_magnitude),
                    'gradient_max': np.max(gradient_magnitude)
                })
            
            # Агрегируем
            grad_means = [ts['gradient_mean'] for ts in texture_stats]
            grad_stds = [ts['gradient_std'] for ts in texture_stats]
            grad_maxs = [ts['gradient_max'] for ts in texture_stats]
            
            return {
                'texture_gradient_mean': np.mean(grad_means),
                'texture_gradient_std': np.mean(grad_stds),
                'texture_gradient_max': np.mean(grad_maxs)
            }
        except Exception as e:
            print(f"Ошибка при извлечении текстурных фичей: {e}")
            return None
    
    def extract_all_features(self, frames):
        """Оптимизированное извлечение всех визуальных фичей"""
        if not frames:
            return {}
        
        features = {}
        
        # CNN фичи (самые важные) - обрабатываем батчами
        cnn_features = self.extract_cnn_features_batch(frames)
        if cnn_features is not None:
            features['cnn_features'] = cnn_features
            features['cnn_mean'] = np.mean(cnn_features, axis=0)
            features['cnn_std'] = np.std(cnn_features, axis=0)
            features['cnn_max'] = np.max(cnn_features, axis=0)
        
        # Быстрые цветовые фичи
        color_features = self.extract_fast_color_features(frames)
        if color_features is not None:
            features.update(color_features)
        
        # Быстрые текстурные фичи
        texture_features = self.extract_fast_texture_features(frames)
        if texture_features is not None:
            features.update(texture_features)
        
        # Упрощенные фичи движения
        motion_features = self.extract_simple_motion_features(frames)
        if motion_features is not None:
            features['motion'] = motion_features
        
        return features


def load_video_frames(video_path, max_frames=20):
    """Оптимизированная загрузка кадров из видео"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Берем меньше кадров для ускорения
        frame_indices = np.linspace(0, total_frames-1, min(max_frames, total_frames), dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Уменьшаем размер кадра для ускорения
                frame = cv2.resize(frame, (320, 240))
                frames.append(frame)
        
        cap.release()
        return frames
    except Exception as e:
        print(f"Ошибка при загрузке кадров из {video_path}: {e}")
        return []


def process_video_visual_features(df_path, output_dir="visual_features", max_frames=20, model_type='mobilenet'):
    """
    Оптимизированная обработка всех видео файлов и извлечение визуальных фичей
    
    Args:
        df_path: Путь к CSV файлу с данными
        output_dir: Директория для сохранения фичей
        max_frames: Максимальное количество кадров для обработки
        model_type: Тип модели для извлечения фичей
    """
    start_time = time.time()
    
    # Создаем директорию
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем данные
    df = pd.read_csv(df_path)
    
    # Инициализируем экстрактор
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Используется устройство: {device}")
    print(f"Модель: {model_type}")
    print(f"Максимум кадров на видео: {max_frames}")
    
    extractor = VisualFeatureExtractor(device=device, model_type=model_type)
    
    visual_features_dict = {}
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Извлечение визуальных фичей"):
        video_path = row['way']
        tag = row['teg']
        part = row['part']
        
        # Создаем уникальный ключ
        video_name = Path(video_path).stem
        key = f"{tag}_{part}_{video_name}"
        
        # Загружаем кадры
        frames = load_video_frames(video_path, max_frames)
        
        if frames:
            # Извлекаем фичи
            features = extractor.extract_all_features(frames)
            if features:
                visual_features_dict[key] = features
    
    # Сохраняем фичи
    output_path = f"{output_dir}/visual_features.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(visual_features_dict, f)
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print(f"\n=== РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ===")
    print(f"Визуальные фичи сохранены в {output_path}")
    print(f"Обработано {len(visual_features_dict)} видео файлов")
    print(f"Время обработки: {processing_time:.2f} секунд")
    print(f"Среднее время на видео: {processing_time/len(visual_features_dict):.2f} секунд")
    
    return visual_features_dict


if __name__ == "__main__":
    # Обрабатываем визуальные фичи с оптимизацией
    visual_features = process_video_visual_features(
        "df.csv", 
        max_frames=20,  # Уменьшили количество кадров
        model_type='mobilenet'  # Более быстрая модель
    )
