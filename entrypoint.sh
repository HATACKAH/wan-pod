#!/bin/bash

# Проверяем наличие токена
if [ -z "$HF_TOKEN" ]; then
    echo "ОШИБКА: Не установлен HF_TOKEN"
    exit 1
fi

# Используем переменную окружения
if [ ! -d "${MODELS_PATH}/Wan2.1-T2V-14B" ] || [ ! -d "${MODELS_PATH}/Wan2.1-I2V-14B-720P" ]; then
    echo "Скачивание моделей в ${MODELS_PATH}..."
    python download_models.py
else
    echo "Модели уже существуют в ${MODELS_PATH}, пропускаем скачивание"
fi

# Запускаем основной обработчик
exec python -u handler.py