# Используем официальный Python образ с минимальным весом
FROM python:3.10-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Копируем файлы с зависимостями и моделью в контейнер
COPY requirements.txt .
COPY best_model.pkl .
COPY app.py .

# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Команда запуска uvicorn с указанием приложения и порта
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
