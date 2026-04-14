#!/bin/bash
# mlflow_server.sh - запуск MLflow tracking server с сохранением данных в локальную папку

# Переменные
MLFLOW_HOST=0.0.0.0
MLFLOW_PORT=5000
BACKEND_STORE_URI="sqlite:///mlflow.db"      # метаданные экспериментов
ARTIFACT_ROOT="./mlruns"                     # артефакты (модели, файлы)

# Запуск сервера
mlflow server \
    --host $MLFLOW_HOST \
    --port $MLFLOW_PORT \
    --backend-store-uri $BACKEND_STORE_URI \
    --default-artifact-root $ARTIFACT_ROOT \
    --serve-artifacts
