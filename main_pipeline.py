#!/usr/bin/env python3
"""
Основной пайплайн для обработки видеозаписей и предсказания насилия
с использованием мультимодального подхода (аудио + видео)
"""

import os
import sys
import argparse
import time
from pathlib import Path
import torch
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pickle
warnings.filterwarnings('ignore')

# Импортируем наши модули
from audio_features import process_video_audio_features, AudioFeatureExtractor
from visual_features import process_video_visual_features, VisualFeatureExtractor
from train_model import train_multimodal_model
from test_model import test_multimodal_model
from multimodal_model import MultimodalViolenceClassifier, create_data_loaders


def check_dependencies():
    """Проверка необходимых зависимостей"""
    required_packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'), 
        ('torchaudio', 'torchaudio'),
        ('librosa', 'librosa'),
        ('opencv-python', 'cv2'),
        ('scikit-learn', 'sklearn'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('tqdm', 'tqdm'),
        ('moviepy', 'moviepy'),
        ('PIL', 'PIL')
    ]
    
    missing_packages = []
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print("❌ Отсутствуют необходимые пакеты:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nУстановите их командой:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ Все необходимые пакеты установлены")
    return True


def check_data_files(df_path):
    """Проверка наличия файлов данных"""
    if not os.path.exists(df_path):
        print(f"❌ Файл данных не найден: {df_path}")
        return False
    
    # Проверяем структуру директорий
    df_dir = Path(df_path).parent
    video_dir = "Movie_Clip_Dataset_A_New_Dataset_for Generalised_Real_World_Violence_Detection"
    
    if not os.path.exists(os.path.join(df_dir, video_dir)):
        print(f"❌ Директория с видео не найдена: {video_dir}")
        return False
    
    print("✅ Файлы данных найдены")
    return True


def extract_features(df_path, max_frames=30, force_extract=False):
    """Извлечение фичей из видео"""
    print("\n" + "="*60)
    print("ШАГ 1: ИЗВЛЕЧЕНИЕ ФИЧЕЙ")
    print("="*60)
    
    audio_features_path = "audio_features/audio_features.pkl"
    visual_features_path = "visual_features/visual_features.pkl"
    
    # Проверяем, нужно ли извлекать фичи
    if not force_extract and os.path.exists(audio_features_path) and os.path.exists(visual_features_path):
        print("✅ Фичи уже извлечены. Используйте --force-extract для переизвлечения.")
        return audio_features_path, visual_features_path
    
    print("🎵 Извлечение аудио фичей...")
    start_time = time.time()
    audio_features = process_video_audio_features(df_path, output_dir="audio_features")
    audio_time = time.time() - start_time
    print(f"⏱️  Аудио фичи извлечены за {audio_time:.2f} секунд")
    
    print("\n🎬 Извлечение визуальных фичей...")
    start_time = time.time()
    visual_features = process_video_visual_features(df_path, output_dir="visual_features", max_frames=max_frames)
    visual_time = time.time() - start_time
    print(f"⏱️  Визуальные фичи извлечены за {visual_time:.2f} секунд")
    
    total_time = audio_time + visual_time
    print(f"\n✅ Все фичи извлечены за {total_time:.2f} секунд")
    
    return audio_features_path, visual_features_path


def train_model(audio_features_path, visual_features_path, df_path, 
                num_epochs=50, batch_size=16, learning_rate=0.001, force_train=False):
    """Обучение модели"""
    print("\n" + "="*60)
    print("ШАГ 2: ОБУЧЕНИЕ МОДЕЛИ")
    print("="*60)
    
    model_path = "models/best_model.pth"
    
    # Проверяем, нужно ли обучать модель
    if not force_train and os.path.exists(model_path):
        print("✅ Модель уже обучена. Используйте --force-train для переобучения.")
        return model_path
    
    print(f"🚀 Начинаем обучение модели...")
    print(f"   - Эпох: {num_epochs}")
    print(f"   - Размер батча: {batch_size}")
    print(f"   - Скорость обучения: {learning_rate}")
    
    start_time = time.time()
    model, history, test_loader = train_multimodal_model(
        audio_features_path=audio_features_path,
        visual_features_path=visual_features_path,
        df_path=df_path,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        save_dir="models"
    )
    train_time = time.time() - start_time
    
    print(f"⏱️  Модель обучена за {train_time:.2f} секунд")
    print(f"✅ Лучшая модель сохранена в {model_path}")
    
    return model_path


def test_model(model_path, audio_features_path, visual_features_path, df_path):
    """Тестирование модели"""
    print("\n" + "="*60)
    print("ШАГ 3: ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("="*60)
    
    scaler_audio_path = "models/scaler_audio.pkl"
    scaler_visual_path = "models/scaler_visual.pkl"
    
    if not os.path.exists(scaler_audio_path) or not os.path.exists(scaler_visual_path):
        print("❌ Скейлеры не найдены. Сначала обучите модель.")
        return None
    
    print("🧪 Тестируем модель на тестовой выборке...")
    start_time = time.time()
    
    metrics, results_df, report = test_multimodal_model(
        model_path=model_path,
        audio_features_path=audio_features_path,
        visual_features_path=visual_features_path,
        df_path=df_path,
        scaler_audio_path=scaler_audio_path,
        scaler_visual_path=scaler_visual_path,
        save_dir="test_results"
    )
    
    test_time = time.time() - start_time
    
    print(f"⏱️  Тестирование завершено за {test_time:.2f} секунд")
    print(f"✅ Результаты сохранены в директории test_results/")
    
    # Выводим основные результаты
    print("\n📊 ОСНОВНЫЕ РЕЗУЛЬТАТЫ:")
    print(f"   - Точность: {metrics['accuracy']:.4f}")
    print(f"   - F1-мера: {metrics['f1_score']:.4f}")
    print(f"   - AUC-ROC: {metrics['auc_roc']:.4f}")
    
    return metrics, results_df, report


def compare_feature_types(df_path, max_frames=30, num_epochs=20, batch_size=16, learning_rate=0.001):
    """Сравнение работы модели с разными типами фичей"""
    print("\n" + "="*60)
    print("СРАВНЕНИЕ ТИПОВ ФИЧЕЙ")
    print("="*60)
    
    # Определяем конфигурации для сравнения
    feature_configs = {
        'audio_only': {'audio': True, 'visual': False, 'name': 'Только аудио'},
        'visual_only': {'audio': False, 'visual': True, 'name': 'Только видео'},
        'multimodal': {'audio': True, 'visual': True, 'name': 'Аудио + видео'}
    }
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for config_name, config in feature_configs.items():
        print(f"\n🔍 Тестируем конфигурацию: {config['name']}")
        
        try:
            # Извлекаем нужные фичи
            audio_features_path = None
            visual_features_path = None
            
            if config['audio']:
                print("   📊 Извлечение аудио фичей...")
                audio_features_path = "audio_features/audio_features.pkl"
                if not os.path.exists(audio_features_path):
                    process_video_audio_features(df_path, output_dir="audio_features")
            
            if config['visual']:
                print("   📊 Извлечение визуальных фичей...")
                visual_features_path = "visual_features/visual_features.pkl"
                if not os.path.exists(visual_features_path):
                    process_video_visual_features(df_path, output_dir="visual_features", max_frames=max_frames)
            
            # Обучаем модель
            print("   🚀 Обучение модели...")
            
            # Создаем временные файлы для случаев, когда один из типов фичей не используется
            temp_audio_path = None
            temp_visual_path = None
            
            # Обучаем модель с правильными путями
            model, history, test_loader = train_multimodal_model(
                audio_features_path=audio_features_path,
                visual_features_path=visual_features_path,
                df_path=df_path,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_dir=f"models/{config_name}"
            )
            
            # Тестируем модель
            print("   🧪 Тестирование модели...")
            model.eval()
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            with torch.no_grad():
                for batch in test_loader:
                    # Правильно обрабатываем None значения
                    if config['audio'] and config['visual']:
                        audio_input = batch['audio'].to(device)
                        visual_input = batch['visual'].to(device)
                    elif config['audio'] and not config['visual']:
                        audio_input = batch['audio'].to(device)
                        visual_input = None
                    elif not config['audio'] and config['visual']:
                        audio_input = None
                        visual_input = batch['visual'].to(device)
                    else:
                        raise ValueError("Хотя бы один тип фичей должен быть активен")
                    
                    labels = batch['label'].to(device)
                    
                    outputs = model(audio_input, visual_input)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            
            # Вычисляем метрики
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            auc = roc_auc_score(all_labels, all_probabilities)
            
            results[config_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc,
                'config': config
            }
            
            print(f"   ✅ {config['name']}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
            
        except Exception as e:
            print(f"   ❌ Ошибка в конфигурации {config['name']}: {e}")
            results[config_name] = {'error': str(e)}
    
    return results


def compare_feature_components(df_path, max_frames=30, num_epochs=20, batch_size=16, learning_rate=0.001):
    """Сравнение работы модели с разными компонентами фичей"""
    print("\n" + "="*60)
    print("СРАВНЕНИЕ КОМПОНЕНТОВ ФИЧЕЙ")
    print("="*60)
    
    # Пока что упрощаем - используем только базовые конфигурации
    # В будущем можно будет добавить кастомные экстракторы
    component_configs = {
        'audio_components': {'audio': True, 'visual': False, 'name': 'Аудио компоненты'},
        'visual_components': {'audio': False, 'visual': True, 'name': 'Визуальные компоненты'},
        'all_components': {'audio': True, 'visual': True, 'name': 'Все компоненты'}
    }
    
    results = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for config_name, config in component_configs.items():
        print(f"\n🔍 Тестируем конфигурацию: {config['name']}")
        
        try:
            # Используем существующую логику из compare_feature_types
            audio_features_path = None
            visual_features_path = None
            
            if config['audio']:
                print("   📊 Использование аудио фичей...")
                audio_features_path = "audio_features/audio_features.pkl"
                if not os.path.exists(audio_features_path):
                    process_video_audio_features(df_path, output_dir="audio_features")
            
            if config['visual']:
                print("   📊 Использование визуальных фичей...")
                visual_features_path = "visual_features/visual_features.pkl"
                if not os.path.exists(visual_features_path):
                    process_video_visual_features(df_path, output_dir="visual_features", max_frames=max_frames)
            
            # Обучаем модель
            print("   🚀 Обучение модели...")
            model, history, test_loader = train_multimodal_model(
                audio_features_path=audio_features_path,
                visual_features_path=visual_features_path,
                df_path=df_path,
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                save_dir=f"models/{config_name}"
            )
            
            # Тестируем модель
            print("   🧪 Тестирование модели...")
            model.eval()
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            with torch.no_grad():
                for batch in test_loader:
                    # Правильно обрабатываем None значения
                    if config['audio'] and config['visual']:
                        audio_input = batch['audio'].to(device)
                        visual_input = batch['visual'].to(device)
                    elif config['audio'] and not config['visual']:
                        audio_input = batch['audio'].to(device)
                        visual_input = None
                    elif not config['audio'] and config['visual']:
                        audio_input = None
                        visual_input = batch['visual'].to(device)
                    else:
                        raise ValueError("Хотя бы один тип фичей должен быть активен")
                    
                    labels = batch['label'].to(device)
                    
                    outputs = model(audio_input, visual_input)
                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            
            # Вычисляем метрики
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='weighted')
            recall = recall_score(all_labels, all_predictions, average='weighted')
            f1 = f1_score(all_labels, all_predictions, average='weighted')
            auc = roc_auc_score(all_labels, all_probabilities)
            
            results[config_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc,
                'config': config
            }
            
            print(f"   ✅ {config['name']}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
            
        except Exception as e:
            print(f"   ❌ Ошибка в конфигурации {config['name']}: {e}")
            results[config_name] = {'error': str(e)}
    
    return results


def extract_custom_audio_features(df_path, extractor, config):
    """Извлечение аудио фичей с заданными компонентами"""
    # Эта функция должна быть реализована для извлечения только нужных компонентов
    # Пока возвращаем None, так как требует модификации AudioFeatureExtractor
    return None


def extract_custom_visual_features(df_path, extractor, config, max_frames):
    """Извлечение визуальных фичей с заданными компонентами"""
    # Эта функция должна быть реализована для извлечения только нужных компонентов
    # Пока возвращаем None, так как требует модификации VisualFeatureExtractor
    return None


def create_custom_data_loaders(audio_features, visual_features, df_path, batch_size):
    """Создание даталоадеров с кастомными фичами"""
    # Эта функция должна быть реализована для работы с кастомными фичами
    return None, None, None


def train_custom_model(model, train_loader, val_loader, device, num_epochs, learning_rate):
    """Обучение кастомной модели"""
    # Эта функция должна быть реализована для обучения с кастомными фичами
    return None


def test_custom_model(model, test_loader, device):
    """Тестирование кастомной модели"""
    # Эта функция должна быть реализована для тестирования с кастомными фичами
    return {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'auc_roc': 0.0}


def create_comparison_heatmap(results, title, save_path):
    """Создание heatmap для сравнения результатов"""
    print(f"\n📊 Создание heatmap: {title}")
    
    # Подготавливаем данные для heatmap
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    configs = list(results.keys())
    
    # Создаем матрицу результатов
    data_matrix = []
    for config in configs:
        if 'error' not in results[config]:
            row = [results[config].get(metric, 0) for metric in metrics]
            data_matrix.append(row)
        else:
            data_matrix.append([0] * len(metrics))
    
    data_matrix = np.array(data_matrix)
    
    # Создаем heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(data_matrix, 
                xticklabels=metrics,
                yticklabels=configs,
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',
                cbar_kws={'label': 'Score'})
    
    plt.title(f'Сравнение результатов: {title}')
    plt.xlabel('Метрики')
    plt.ylabel('Конфигурации')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Сохраняем heatmap
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Heatmap сохранена: {save_path}")


def find_best_model(results):
    """Поиск лучшей модели по F1-score"""
    best_config = None
    best_f1 = 0
    
    for config_name, result in results.items():
        if 'error' not in result and result.get('f1_score', 0) > best_f1:
            best_f1 = result['f1_score']
            best_config = config_name
    
    return best_config, best_f1


def save_best_model(best_config, results, save_dir="best_models"):
    """Сохранение лучшей модели"""
    print(f"\n💾 Сохранение лучшей модели: {best_config}")
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Сохраняем результаты
    results_path = os.path.join(save_dir, "comparison_results.pkl")
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Сохраняем информацию о лучшей модели
    best_info = {
        'best_config': best_config,
        'best_f1_score': results[best_config]['f1_score'],
        'all_metrics': results[best_config]
    }
    
    best_info_path = os.path.join(save_dir, "best_model_info.pkl")
    with open(best_info_path, 'wb') as f:
        pickle.dump(best_info, f)
    
    print(f"✅ Лучшая модель сохранена в {save_dir}/")
    print(f"   - Конфигурация: {best_config}")
    print(f"   - F1-score: {results[best_config]['f1_score']:.4f}")
    
    return best_info


def features_comparison_mode(df_path, max_frames=30, num_epochs=20, batch_size=16, learning_rate=0.001):
    """Основная функция режима сравнения фичей"""
    print("\n" + "="*60)
    print("РЕЖИМ СРАВНЕНИЯ ФИЧЕЙ")
    print("="*60)
    
    start_time = time.time()
    
    # 1. Сравнение типов фичей
    print("\n🔍 ЭТАП 1: Сравнение типов фичей")
    feature_type_results = compare_feature_types(df_path, max_frames, num_epochs, batch_size, learning_rate)
    
    # 2. Сравнение компонентов фичей
    print("\n🔍 ЭТАП 2: Сравнение компонентов фичей")
    component_results = compare_feature_components(df_path, max_frames, num_epochs, batch_size, learning_rate)
    
    # 3. Создание heatmap
    print("\n📊 ЭТАП 3: Создание визуализаций")
    create_comparison_heatmap(feature_type_results, "Сравнение типов фичей", "comparison_results/feature_types_heatmap.png")
    create_comparison_heatmap(component_results, "Сравнение компонентов фичей", "comparison_results/feature_components_heatmap.png")
    
    # 4. Поиск и сохранение лучшей модели
    print("\n🏆 ЭТАП 4: Поиск лучшей модели")
    
    # Объединяем все результаты
    all_results = {**feature_type_results, **component_results}
    
    best_config, best_f1 = find_best_model(all_results)
    if best_config:
        best_info = save_best_model(best_config, all_results)
        
        print(f"\n🎉 ЛУЧШАЯ МОДЕЛЬ НАЙДЕНА!")
        print(f"   - Конфигурация: {best_config}")
        print(f"   - F1-score: {best_f1:.4f}")
        print(f"   - Accuracy: {all_results[best_config]['accuracy']:.4f}")
        print(f"   - AUC-ROC: {all_results[best_config]['auc_roc']:.4f}")
    else:
        print("❌ Не удалось найти лучшую модель")
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Общее время выполнения: {total_time:.2f} секунд")
    
    return all_results, best_config


def main():
    """Основная функция"""
    parser = argparse.ArgumentParser(description='Мультимодальная классификация насилия в видео')
    
    # Основные параметры
    parser.add_argument('--df-path', type=str, default='df.csv',
                       help='Путь к CSV файлу с данными')
    parser.add_argument('--max-frames', type=int, default=30,
                       help='Максимальное количество кадров для обработки')
    
    # Параметры обучения
    parser.add_argument('--epochs', type=int, default=50,
                       help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Размер батча')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Скорость обучения')
    
    # Флаги
    parser.add_argument('--force-extract', action='store_true',
                       help='Принудительное извлечение фичей')
    parser.add_argument('--force-train', action='store_true',
                       help='Принудительное обучение модели')
    parser.add_argument('--skip-extract', action='store_true',
                       help='Пропустить извлечение фичей')
    parser.add_argument('--skip-train', action='store_true',
                       help='Пропустить обучение модели')
    parser.add_argument('--skip-test', action='store_true',
                       help='Пропустить тестирование модели')
    
    # Режимы работы
    parser.add_argument('--extract-only', action='store_true',
                       help='Только извлечение фичей')
    parser.add_argument('--train-only', action='store_true',
                       help='Только обучение модели')
    parser.add_argument('--test-only', action='store_true',
                       help='Только тестирование модели')
    parser.add_argument('--features-comp', action='store_true',
                       help='Режим сравнения фичей (сравнивает разные типы и компоненты фичей)')
    
    args = parser.parse_args()
    
    print("🎬 МУЛЬТИМОДАЛЬНАЯ КЛАССИФИКАЦИЯ НАСИЛИЯ В ВИДЕО")
    print("="*60)
    print(f"📁 Данные: {args.df_path}")
    print(f"🎞️  Максимум кадров: {args.max_frames}")
    print(f"🚀 Эпох обучения: {args.epochs}")
    print(f"📦 Размер батча: {args.batch_size}")
    print(f"📈 Скорость обучения: {args.learning_rate}")
    print("="*60)
    
    # Проверяем зависимости
    if not check_dependencies():
        sys.exit(1)
    
    # Проверяем данные
    if not check_data_files(args.df_path):
        sys.exit(1)
    
    # Проверяем доступность GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Используется устройство: {device}")
    
    start_time = time.time()
    
    try:
        # Режим сравнения фичей
        if args.features_comp:
            print("🔍 ЗАПУСК РЕЖИМА СРАВНЕНИЯ ФИЧЕЙ")
            results, best_config = features_comparison_mode(
                args.df_path, args.max_frames, args.epochs, args.batch_size, args.learning_rate
            )
            return
        
        # Шаг 1: Извлечение фичей
        if not args.skip_extract and not args.train_only and not args.test_only:
            audio_features_path, visual_features_path = extract_features(
                args.df_path, args.max_frames, args.force_extract
            )
        else:
            audio_features_path = "audio_features/audio_features.pkl"
            visual_features_path = "visual_features/visual_features.pkl"
        
        if args.extract_only:
            print("✅ Извлечение фичей завершено")
            return
        
        # Шаг 2: Обучение модели
        if not args.skip_train and not args.extract_only and not args.test_only:
            model_path = train_model(
                audio_features_path, visual_features_path, args.df_path,
                args.epochs, args.batch_size, args.learning_rate, args.force_train
            )
        else:
            model_path = "models/best_model.pth"
        
        if args.train_only:
            print("✅ Обучение модели завершено")
            return
        
        # Шаг 3: Тестирование модели
        if not args.skip_test and not args.extract_only and not args.train_only:
            test_model(model_path, audio_features_path, visual_features_path, args.df_path)
        
        total_time = time.time() - start_time
        print(f"\n🎉 ВСЕ ЭТАПЫ ЗАВЕРШЕНЫ ЗА {total_time:.2f} СЕКУНД")
        
        # Выводим информацию о результатах
        print("\n📁 РЕЗУЛЬТАТЫ СОХРАНЕНЫ В:")
        print("   - audio_features/ - аудио фичи")
        print("   - visual_features/ - визуальные фичи")
        print("   - models/ - обученная модель")
        print("   - test_results/ - результаты тестирования")
        
    except KeyboardInterrupt:
        print("\n❌ Процесс прерван пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
