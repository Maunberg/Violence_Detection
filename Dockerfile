# Используем официальный Python образ
FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gradio==3.50.2

# Копируем весь проект
COPY . .

# Создаем директорию для временных файлов
RUN mkdir -p /tmp/violence_detection

# Устанавливаем переменные окружения
ENV PYTHONPATH=/app
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7861

# Открываем порт для Gradio
EXPOSE 7861

# Команда по умолчанию
CMD ["python", "gradio_app.py"]