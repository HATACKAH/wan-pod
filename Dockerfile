FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

# Предотвращаем интерактивные запросы при установке пакетов
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Установка переменных окружения для CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Копирование файлов проекта
COPY . /app
WORKDIR /app

# Установка зависимостей Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Создание директорий
RUN mkdir -p /tmp

# Установка переменных окружения
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Копируем и делаем исполняемым скрипт запуска
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Используем entrypoint.sh как точку входа
ENTRYPOINT ["/app/entrypoint.sh"]