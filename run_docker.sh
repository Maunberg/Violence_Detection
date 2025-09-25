#!/bin/bash

# Скрипт для запуска Violence Detection в Docker

echo "🎬 Запуск Violence Detection в Docker..."

# Проверяем наличие Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker не установлен. Установите Docker и попробуйте снова."
    exit 1
fi

# Проверяем наличие docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose не установлен. Установите docker-compose и попробуйте снова."
    exit 1
fi

# Создаем директорию для временных файлов
mkdir -p temp

# Останавливаем существующие контейнеры
echo "🛑 Остановка существующих контейнеров..."
sudo docker-compose down

# Собираем образ
echo "🔨 Сборка Docker образа..."
sudo docker-compose build

# Запускаем контейнер
echo "🚀 Запуск контейнера..."
sudo docker-compose up -d

# Ждем запуска
echo "⏳ Ожидание запуска приложения..."
sleep 10

# Проверяем статус
if sudo docker-compose ps | grep -q "Up"; then
    echo "✅ Приложение успешно запущено!"
    echo "🌐 Откройте браузер и перейдите по адресу: http://localhost:7861"
    echo "📊 Для просмотра логов используйте: sudo docker-compose logs -f"
    echo "🛑 Для остановки используйте: sudo docker-compose down"
else
    echo "❌ Ошибка при запуске приложения"
    echo "📋 Логи:"
    sudo docker-compose logs
    exit 1
fi
