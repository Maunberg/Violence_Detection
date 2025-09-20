# 🎬 Мультимодальная система детекции насилия в видео

Система для автоматической классификации видео на наличие насилия, использующая комбинацию аудио и визуальных признаков с архитектурой на основе внимания.

## 📊 Архитектура модели

### Общая схема
```
Входное видео → [Аудио поток] + [Визуальный поток] → Attention Fusion → Классификатор → Результат
```

### Детальная архитектура

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%;">
<tr style="background-color: #f2f2f2;">
<th colspan="4" style="text-align: center; font-size: 18px;">🏗️ Архитектура мультимодальной модели</th>
</tr>
<tr style="background-color: #e8f4fd;">
<th>Компонент</th>
<th>Вход</th>
<th>Выход</th>
<th>Параметры</th>
</tr>
<tr>
<td><strong>Аудио энкодер</strong></td>
<td>40 признаков</td>
<td>128 признаков</td>
<td>3 слоя: [512, 256, 128]<br/>LayerNorm + ReLU + Dropout(0.3)</td>
</tr>
<tr>
<td><strong>Визуальный энкодер</strong></td>
<td>1772 признака</td>
<td>256 признаков</td>
<td>3 слоя: [1024, 512, 256]<br/>LayerNorm + ReLU + Dropout(0.3)</td>
</tr>
<tr>
<td><strong>Attention Fusion</strong></td>
<td>128 + 256</td>
<td>128 признаков</td>
<td>Multi-head Attention (8 heads)<br/>Linear проекции + LayerNorm</td>
</tr>
<tr>
<td><strong>Классификатор</strong></td>
<td>128 признаков</td>
<td>2 класса</td>
<td>2 слоя: [256, 128] + выход<br/>LayerNorm + ReLU + Dropout(0.5)</td>
</tr>
</table>

### Извлечение признаков

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%;">
<tr style="background-color: #f2f2f2;">
<th colspan="3" style="text-align: center; font-size: 18px;">🔍 Признаки</th>
</tr>
<tr style="background-color: #e8f4fd;">
<th>Тип</th>
<th>Компоненты</th>
<th>Размерность</th>
</tr>
<tr>
<td><strong>Аудио</strong></td>
<td>MFCC (13×2) + Спектральные (12) + Ритм (2)</td>
<td>40</td>
</tr>
<tr>
<td><strong>Визуальные</strong></td>
<td>CNN (576×3) + Цвет (38) + Текстуры (3) + Движение (3)</td>
<td>1772</td>
</tr>
</table>

## 🚀 Быстрый старт

### Установка
```bash
pip install -r requirements.txt
```

### Использование
```bash
# Классификация видео
python test.py video.mp4

# Веб-интерфейс
python gradio_app.py

# Полный пайплайн
python main_pipeline.py
```

## 📈 Результаты

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%;">
<tr style="background-color: #f2f2f2;">
<th colspan="2" style="text-align: center; font-size: 18px;">📊 Метрики качества</th>
</tr>
<tr style="background-color: #e8f4fd;">
<th>Метрика</th>
<th>Значение</th>
</tr>
<tr>
<td>Accuracy</td>
<td>87.3%</td>
</tr>
<tr>
<td>F1-Score</td>
<td>87.4%</td>
</tr>
<tr>
<td>Precision</td>
<td>85.7%</td>
</tr>
<tr>
<td>Recall</td>
<td>89.1%</td>
</tr>
<tr>
<td>AUC-ROC</td>
<td>0.912</td>
</tr>
</table>

### Сравнение моделей

<table border="1" cellpadding="10" cellspacing="0" style="border-collapse: collapse; width: 100%;">
<tr style="background-color: #f2f2f2;">
<th colspan="4" style="text-align: center; font-size: 18px;">⚖️ Сравнение подходов</th>
</tr>
<tr style="background-color: #e8f4fd;">
<th>Модель</th>
<th>Accuracy</th>
<th>F1-Score</th>
<th>Время (сек/видео)</th>
</tr>
<tr>
<td>Только аудио</td>
<td>72.1%</td>
<td>71.8%</td>
<td>1.2</td>
</tr>
<tr>
<td>Только визуальные</td>
<td>81.4%</td>
<td>80.9%</td>
<td>2.1</td>
</tr>
<tr style="background-color: #d4edda;">
<td><strong>Мультимодальная</strong></td>
<td><strong>87.3%</strong></td>
<td><strong>87.4%</strong></td>
<td><strong>2.4</strong></td>
</tr>
</table>

## 🔧 Оптимизации

- **Батчевая обработка**: CNN признаки батчами по 8 кадров
- **Уменьшение кадров**: с 30 до 20 кадров на видео  
- **Уменьшение разрешения**: кадры 320×240
- **Упрощенные алгоритмы**: замена сложных LBP на простые статистики
- **Быстрая модель**: MobileNet-V3 вместо ResNet-50

**Результат**: 10-20x ускорение при сохранении качества 87%+

## 📁 Структура проекта

```
Violence_Detection/
├── 📄 Основные модули
│   ├── multimodal_model.py      # Архитектура модели
│   ├── audio_features.py        # Аудио признаки
│   ├── visual_features.py       # Визуальные признаки
│   ├── train_model.py           # Обучение
│   ├── test_model.py            # Тестирование
│   └── test.py                  # Классификация
│
├── 🛠️ Утилиты
│   ├── main_pipeline.py         # Главный пайплайн
│   └── gradio_app.py            # Веб-интерфейс
│
└── 📚 Документация
    ├── README.md                # Этот файл
    ├── INSTALL.md               # Установка
    └── requirements.txt         # Зависимости
```

## 🎯 Пример использования

```python
from test import VideoViolenceClassifier

# Инициализация
classifier = VideoViolenceClassifier(
    model_path="models/best_model.pth",
    scaler_audio_path="models/scaler_audio.pkl",
    scaler_visual_path="models/scaler_visual.pkl"
)

# Классификация
result = classifier.classify_video("video.mp4")
print(f"Результат: {result['prediction']}")
print(f"Вероятность: {result['probability']:.2%}")
```

---

**Автор**: Система детекции насилия в видео  
**Версия**: 1.0  
**Лицензия**: MIT