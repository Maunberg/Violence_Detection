#!/usr/bin/env python3
"""
Gradio интерфейс для классификации видео на наличие насилия
Поддерживает загрузку видео и отображение результата с графиком жестокости
"""

import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Импортируем наш классификатор
from test import VideoViolenceClassifier


class GradioViolenceClassifier:
    """Класс для работы с Gradio интерфейсом"""
    
    def __init__(self, model_path, scaler_audio_path, scaler_visual_path, device='auto'):
        """
        Инициализация классификатора для Gradio
        
        Args:
            model_path: Путь к обученной модели
            scaler_audio_path: Путь к скейлеру аудио фичей
            scaler_visual_path: Путь к скейлеру визуальных фичей
            device: Устройство для вычислений
        """
        self.classifier = VideoViolenceClassifier(
            model_path=model_path,
            scaler_audio_path=scaler_audio_path,
            scaler_visual_path=scaler_visual_path,
            device=device
        )
        
        # Настройки matplotlib для русского языка
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Liberation Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
    
    def classify_video_simple(self, video_file):
        """
        Простая классификация видео
        
        Args:
            video_file: Загруженный видео файл
            
        Returns:
            tuple: (результат, None) - для совместимости с Gradio
        """
        if video_file is None:
            return "Пожалуйста, загрузите видео файл", None
        
        try:
            # Получаем путь к файлу
            video_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
            
            # Классифицируем видео
            class_name, probabilities = self.classifier.classify_video(
                video_path, return_probabilities=True
            )
            
            # Формируем результат
            result = f"**Результат классификации: {class_name}**\n\n"
            result += f"Вероятности:\n"
            result += f"- Non-Violent: {probabilities[0]:.4f} ({probabilities[0]*100:.1f}%)\n"
            result += f"- Violent: {probabilities[1]:.4f} ({probabilities[1]*100:.1f}%)\n\n"
            
            # Добавляем интерпретацию
            if probabilities[1] > 0.7:
                result += "🔴 **Высокая вероятность насилия**"
            elif probabilities[1] > 0.5:
                result += "🟡 **Средняя вероятность насилия**"
            else:
                result += "🟢 **Низкая вероятность насилия**"
            
            return result, None
            
        except Exception as e:
            return f"Ошибка при обработке видео: {str(e)}", None
    
    def classify_video_with_graph(self, video_file, batch_size, overlap):
        """
        Классификация видео с построением графика жестокости
        
        Args:
            video_file: Загруженный видео файл
            batch_size: Размер батча (количество кадров)
            overlap: Перекрытие между батчами
            
        Returns:
            tuple: (результат, график)
        """
        if video_file is None:
            return "Пожалуйста, загрузите видео файл", None
        
        try:
            # Получаем путь к файлу
            video_path = video_file.name if hasattr(video_file, 'name') else str(video_file)
            
            # Классифицируем видео по батчам
            overall_class, batch_probs, time_stamps = self.classifier.classify_video_batch(
                video_path, batch_size=batch_size, overlap=overlap
            )
            
            # Формируем текстовый результат
            result = f"**Результат классификации: {overall_class}**\n\n"
            result += f"Статистика по времени:\n"
            result += f"- Средняя вероятность насилия: {np.mean(batch_probs):.4f}\n"
            result += f"- Максимальная вероятность: {np.max(batch_probs):.4f}\n"
            result += f"- Минимальная вероятность: {np.min(batch_probs):.4f}\n"
            result += f"- Стандартное отклонение: {np.std(batch_probs):.4f}\n\n"
            
            # Добавляем интерпретацию
            avg_prob = np.mean(batch_probs)
            if avg_prob > 0.7:
                result += "🔴 **Высокая вероятность насилия в видео**"
            elif avg_prob > 0.5:
                result += "🟡 **Средняя вероятность насилия в видео**"
            else:
                result += "🟢 **Низкая вероятность насилия в видео**"
            
            # Строим график
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Основной график
            ax.plot(time_stamps, batch_probs, 'b-', linewidth=2, label='Вероятность насилия')
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Порог классификации')
            
            # Заливка областей
            ax.fill_between(time_stamps, batch_probs, 0.5, 
                           where=np.array(batch_probs) >= 0.5, 
                           color='red', alpha=0.2, label='Зона насилия')
            ax.fill_between(time_stamps, batch_probs, 0.5, 
                           where=np.array(batch_probs) < 0.5, 
                           color='green', alpha=0.2, label='Зона безопасности')
            
            # Настройки графика
            ax.set_xlabel('Время (секунды)', fontsize=12)
            ax.set_ylabel('Вероятность насилия', fontsize=12)
            ax.set_title('График жестокости в видео', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            # Добавляем аннотации для пиковых значений
            max_idx = np.argmax(batch_probs)
            min_idx = np.argmin(batch_probs)
            
            ax.annotate(f'Макс: {batch_probs[max_idx]:.3f}', 
                       xy=(time_stamps[max_idx], batch_probs[max_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            ax.annotate(f'Мин: {batch_probs[min_idx]:.3f}', 
                       xy=(time_stamps[min_idx], batch_probs[min_idx]),
                       xytext=(10, -20), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='green', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.tight_layout()
            
            return result, fig
            
        except Exception as e:
            return f"Ошибка при обработке видео: {str(e)}", None


def create_gradio_interface():
    """Создание Gradio интерфейса"""
    
    # Проверяем существование файлов модели
    model_path = "models/best_model.pth"
    scaler_audio_path = "models/scaler_audio.pkl"
    scaler_visual_path = "models/scaler_visual.pkl"
    
    if not all(os.path.exists(path) for path in [model_path, scaler_audio_path, scaler_visual_path]):
        print("Ошибка: Не найдены файлы модели или скейлеров")
        print("Убедитесь, что файлы находятся в правильных путях:")
        print(f"- {model_path}")
        print(f"- {scaler_audio_path}")
        print(f"- {scaler_visual_path}")
        return None
    
    # Инициализируем классификатор
    try:
        gradio_classifier = GradioViolenceClassifier(
            model_path=model_path,
            scaler_audio_path=scaler_audio_path,
            scaler_visual_path=scaler_visual_path,
            device='auto'
        )
        print("Классификатор инициализирован успешно")
    except Exception as e:
        print(f"Ошибка инициализации классификатора: {e}")
        return None
    
    # Создаем интерфейс
    with gr.Blocks(
        title="Детектор насилия в видео",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            margin-bottom: 30px;
        }
        .result-box {
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
        }
        """
    ) as interface:
        
        # Заголовок
        gr.HTML("""
        <div class="main-header">
            <h1>🎬 Детектор насилия в видео</h1>
            <p>Загрузите видео для анализа на наличие насилия с помощью мультимодальной нейронной сети</p>
        </div>
        """)
        
        with gr.Tabs():
            # Вкладка простой классификации
            with gr.Tab("🔍 Простая классификация"):
                gr.Markdown("""
                ### Простая классификация видео
                Загрузите видео файл для получения результата классификации на Violent/Non-Violent
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        simple_video = gr.File(
                            label="Загрузить видео",
                            file_types=["video"]
                        )
                        simple_btn = gr.Button("Классифицировать", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        simple_result = gr.Markdown(
                            label="Результат",
                            value="Загрузите видео и нажмите кнопку для классификации"
                        )
                
                simple_btn.click(
                    fn=gradio_classifier.classify_video_simple,
                    inputs=[simple_video],
                    outputs=[simple_result, gr.Plot(visible=False)]
                )
            
            # Вкладка анализа с графиком
            with gr.Tab("📊 Анализ с графиком"):
                gr.Markdown("""
                ### Детальный анализ с графиком жестокости
                Получите подробный анализ видео с построением графика вероятности насилия по времени
                """)
                
                with gr.Row():
                    with gr.Column(scale=1):
                        graph_video = gr.File(
                            label="Загрузить видео",
                            file_types=["video"]
                        )
                        
                        with gr.Group():
                            gr.Markdown("**Параметры анализа:**")
                            batch_size = gr.Slider(
                                minimum=5,
                                maximum=50,
                                value=15,
                                step=1,
                                label="Размер батча (кадры)",
                                info="Количество кадров для анализа в одном батче"
                            )
                            overlap = gr.Slider(
                                minimum=0.0,
                                maximum=0.9,
                                value=0.3,
                                step=0.1,
                                label="Перекрытие батчей",
                                info="Степень перекрытия между соседними батчами"
                            )
                        
                        graph_btn = gr.Button("Анализировать с графиком", variant="primary", size="lg")
                    
                    with gr.Column(scale=2):
                        graph_result = gr.Markdown(
                            label="Результат анализа",
                            value="Загрузите видео и настройте параметры для детального анализа"
                        )
                        graph_plot = gr.Plot(
                            label="График жестокости",
                            value=None
                        )
                
                graph_btn.click(
                    fn=gradio_classifier.classify_video_with_graph,
                    inputs=[graph_video, batch_size, overlap],
                    outputs=[graph_result, graph_plot]
                )
        
        # Информационная секция
        with gr.Accordion("ℹ️ Информация о модели", open=False):
            gr.Markdown("""
            ### О модели
            
            **Мультимодальная нейронная сеть** для детекции насилия в видео использует:
            
            - **Аудио фичи**: MFCC коэффициенты, спектральные характеристики, ритмические паттерны
            - **Визуальные фичи**: CNN фичи (ResNet50), цветовые характеристики, текстуры, движение
            - **Архитектура**: Attention-based fusion для объединения модальностей
            
            **Метрики модели:**
            - Точность: 88.8%
            - F1-мера: 88.8%
            - AUC-ROC: 94.2%
            
            **Поддерживаемые форматы видео:**
            - MP4, AVI, MOV, MKV и другие форматы, поддерживаемые OpenCV
            
            **Рекомендации:**
            - Видео должно содержать аудио дорожку
            - Оптимальная длительность: 10-60 секунд
            - Для лучших результатов используйте видео с четким изображением и звуком
            """)
        
        # Примеры использования
        with gr.Accordion("📝 Примеры использования", open=False):
            gr.Markdown("""
            ### Как использовать интерфейс
            
            1. **Простая классификация:**
               - Загрузите видео файл
               - Нажмите "Классифицировать"
               - Получите результат: Violent или Non-Violent с вероятностями
            
            2. **Анализ с графиком:**
               - Загрузите видео файл
               - Настройте параметры анализа (размер батча, перекрытие)
               - Нажмите "Анализировать с графиком"
               - Получите детальный анализ с графиком жестокости по времени
            
            ### Интерпретация результатов
            
            - **🟢 Низкая вероятность (0-0.5)**: Видео скорее всего не содержит насилия
            - **🟡 Средняя вероятность (0.5-0.7)**: Неопределенный результат, требует дополнительного анализа
            - **🔴 Высокая вероятность (0.7-1.0)**: Высокая вероятность наличия насилия в видео
            """)
    
    return interface


def main():
    """Основная функция для запуска Gradio приложения"""
    print("Запуск Gradio интерфейса для детекции насилия в видео...")
    
    # Создаем интерфейс
    interface = create_gradio_interface()
    
    if interface is None:
        print("Не удалось создать интерфейс. Проверьте наличие файлов модели.")
        sys.exit(1)
    
    # Запускаем интерфейс
    try:
        interface.launch(
            server_name="0.0.0.0",  # Доступно для всех IP
            server_port=7861,       # Порт (изменен на 7861)
            share=True,             # Cоздавать публичную ссылку
            debug=False,            # Отключить debug режим
            show_error=True,        # Показывать ошибки
            quiet=False             # Показывать логи
        )
    except Exception as e:
        print(f"Ошибка при запуске интерфейса: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
