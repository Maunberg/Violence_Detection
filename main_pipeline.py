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
warnings.filterwarnings('ignore')

# Импортируем наши модули
from audio_features import process_video_audio_features
from visual_features import process_video_visual_features
from train_model import train_multimodal_model
from test_model import test_multimodal_model


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
