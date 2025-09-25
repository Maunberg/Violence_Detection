#!/bin/bash

# Быстрый запуск без пересборки (использует существующий образ)

echo "🚀 Быстрый запуск Violence Detection (порт 7862)..."

# Останавливаем существующие контейнеры
echo "🛑 Остановка существующих контейнеров..."
sudo docker-compose down 2>/dev/null || true

# Запускаем контейнер (используем существующий образ)
echo "🚀 Запуск контейнера на порту 7862..."
sudo docker-compose up -d

# Ждем запуска
echo "⏳ Ожидание запуска приложения..."
sleep 10

# Проверяем статус
if sudo docker-compose ps | grep -q "Up"; then
    echo "✅ Приложение успешно запущено!"
    echo "🌐 Откройте браузер и перейдите по адресу: http://localhost:7862"
    echo ""
    echo "📊 Для просмотра логов:"
    echo "   sudo docker-compose logs -f violence-detection"
    echo ""
    echo "🔍 Для поиска внешней ссылки:"
    echo "   sudo docker-compose logs violence-detection | grep 'Running on'"
    echo ""
    echo "🛑 Для остановки: sudo docker-compose down"
else
    echo "❌ Ошибка при запуске приложения"
    echo "📋 Логи:"
    sudo docker-compose logs
    exit 1
fi
