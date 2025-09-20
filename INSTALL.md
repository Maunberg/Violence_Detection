# Инструкция по установке

## Системные требования

- Python 3.7 или выше
- 8+ GB RAM (рекомендуется 16+ GB)
- 10+ GB свободного места на диске
- CUDA-совместимая видеокарта (опционально, для ускорения)

## Установка зависимостей

### 1. Создание виртуального окружения (рекомендуется)

```bash
# Создание виртуального окружения
python -m venv violence_detection_env

# Активация (Linux/Mac)
source violence_detection_env/bin/activate

# Активация (Windows)
violence_detection_env\Scripts\activate
```

### 2. Установка Python пакетов

```bash
# Обновление pip
pip install --upgrade pip

# Установка зависимостей
pip install -r requirements.txt
```

### 3. Установка FFmpeg

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg
```

#### CentOS/RHEL:
```bash
sudo yum install ffmpeg
```

#### macOS:
```bash
brew install ffmpeg
```

#### Windows:
1. Скачайте FFmpeg с https://ffmpeg.org/download.html
2. Распакуйте в папку (например, `C:\ffmpeg`)
3. Добавьте `C:\ffmpeg\bin` в PATH

### 4. Проверка установки

```bash
# Проверка Python пакетов
python -c "import torch, torchvision, librosa, cv2, sklearn; print('Все пакеты установлены успешно')"

# Проверка FFmpeg
ffmpeg -version
```

## Установка CUDA (опционально)

Для ускорения обучения на GPU:

### 1. Проверка поддержки CUDA
```bash
nvidia-smi
```

### 2. Установка PyTorch с поддержкой CUDA
```bash
# Для CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Для CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. Проверка CUDA
```bash
python -c "import torch; print(f'CUDA доступна: {torch.cuda.is_available()}')"
```

## Подготовка данных

### 1. Структура директорий
Убедитесь, что ваши данные организованы следующим образом:

```
project_directory/
├── df.csv
└── Movie_Clip_Dataset_A_New_Dataset_for Generalised_Real_World_Violence_Detection/
    ├── train/
    │   ├── violent/
    │   │   ├── video1.mp4
    │   │   └── video2.mp4
    │   └── non_violent/
    │       ├── video3.mp4
    │       └── video4.mp4
    ├── val/
    │   ├── violent/
    │   └── non_violent/
    └── test/
        ├── violent/
        └── non_violent/
```

### 2. Формат CSV файла
Файл `df.csv` должен содержать колонки:
- `way` - путь к видео файлу
- `part` - часть датасета (train/val/test)
- `teg` - метка (violent/non_violent)
- `duration` - длительность видео в секундах

Пример:
```csv
way,part,teg,duration
Movie_Clip_Dataset_A_New_Dataset_for Generalised_Real_World_Violence_Detection/train/violent/video1.mp4,train,violent,5.123
Movie_Clip_Dataset_A_New_Dataset_for Generalised_Real_World_Violence_Detection/train/non_violent/video2.mp4,train,non_violent,4.567
```

## Быстрый тест

После установки выполните быстрый тест:

```bash
# Демонстрация на небольшом датасете
python demo.py
```

## Возможные проблемы

### 1. Ошибка "ffmpeg not found"
- Убедитесь, что FFmpeg установлен и добавлен в PATH
- Перезапустите терминал после установки

### 2. Ошибка "CUDA out of memory"
- Уменьшите размер батча: `--batch-size 8`
- Уменьшите количество кадров: `--max-frames 15`

### 3. Ошибка "No module named 'torch'"
- Убедитесь, что виртуальное окружение активировано
- Переустановите PyTorch: `pip install torch torchvision torchaudio`

### 4. Медленная обработка
- Используйте GPU: установите CUDA версию PyTorch
- Уменьшите количество кадров: `--max-frames 20`
- Используйте меньший размер батча

## Производительность

### Рекомендуемые настройки:

**Для CPU:**
```bash
python main_pipeline.py --batch-size 8 --max-frames 20 --epochs 30
```

**Для GPU:**
```bash
python main_pipeline.py --batch-size 16 --max-frames 30 --epochs 50
```

### Время выполнения (примерное):

- **Извлечение фичей**: 1-2 минуты на видео
- **Обучение**: 10-30 минут (зависит от размера датасета)
- **Тестирование**: 1-2 минуты

## Поддержка

При возникновении проблем:
1. Проверьте версии всех пакетов: `pip list`
2. Убедитесь в правильности структуры данных
3. Проверьте логи ошибок в консоли
4. Попробуйте запустить демо: `python demo.py`
