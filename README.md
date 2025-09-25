# 🎬 Система детекции насилия в видео

Мультимодальная нейронная сеть для классификации видео на наличие насильственного контента с использованием аудио и визуальных признаков.

## 🎯 Описание

Система анализирует видео, извлекая аудио и визуальные признаки, и использует мультимодальную архитектуру с attention mechanism для классификации на классы "Violent" и "Non-Violent".

### Архитектура

- **Аудио признаки**: MFCC коэффициенты, спектральные характеристики, ритмические признаки (40 измерений)
- **Визуальные признаки**: CNN признаки (MobileNet-V3), цветовые характеристики, текстуры, движение (1772 измерения)
- **Модель**: Мультимодальная сеть с attention fusion и классификатором

## 📊 Результаты

| Метрика | Значение |
|---------|----------|
| **Accuracy** | **86.2%** |
| **F1-Score** | **86.2%** |
| **AUC-ROC** | **0.922** |

## 🚀 Быстрый старт

### Установка

```bash
# Установка зависимостей
pip install -r Violence_Detection/requirements.txt
```

### Использование

```bash
# Полный пайплайн (извлечение + обучение + тестирование)
cd Violence_Detection
python main_pipeline.py

# Классификация видео
python test.py path/to/video.mp4

# Веб-интерфейс
python gradio_app.py
```

### Программное использование

```python
from Violence_Detection.test import VideoViolenceClassifier

classifier = VideoViolenceClassifier(
    model_path="Violence_Detection/models/best_model.pth",
    scaler_audio_path="Violence_Detection/models/scaler_audio.pkl",
    scaler_visual_path="Violence_Detection/models/scaler_visual.pkl"
)

result = classifier.classify_video("sample_video.mp4")
print(f"Результат: {result['prediction']}")
print(f"Вероятность: {result['probability']:.2%}")
```

## 📁 Структура проекта

```
Violence_Detection/
├── multimodal_model.py      # Архитектура модели
├── train_model.py           # Обучение
├── test_model.py            # Тестирование
├── audio_features.py        # Аудио признаки
├── visual_features.py       # Визуальные признаки
├── main_pipeline.py         # Главный пайплайн
├── test.py                  # Классификация видео
├── gradio_app.py            # Веб-интерфейс
├── models/                  # Обученные модели
└── test_results/            # Результаты тестирования
```

## 🛠️ Технические детали

- **Python**: 3.8+
- **PyTorch**: 1.9+
- **Оптимизатор**: Adam (lr=0.001)
- **Batch size**: 32
- **Epochs**: 50 с early stopping
- **Нормализация**: StandardScaler для всех признаков