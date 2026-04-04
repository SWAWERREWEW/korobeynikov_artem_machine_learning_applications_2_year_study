import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from pathlib import Path
import requests
import os
import mlflow

# Импортируем функцию обучения
from train_model import train

def download_data():
    """Загрузка датасета cardata.csv"""
    # Если файл уже существует локально — используем его
    if os.path.exists("cardata.csv"):
        print("✅ Файл cardata.csv уже существует")
        return True
    
    # Попытка скачать из репозитория (замените URL на актуальный)
    url = "https://raw.githubusercontent.com/your-repo/cardata.csv"  # 🔁 Замените на свой URL
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open("cardata.csv", "wb") as f:
            f.write(response.content)
        print(f"✅ Данные загружены: cardata.csv")
        return True
    except Exception as e:
        print(f"⚠️ Не удалось скачать данные: {e}")
        print("💡 Убедитесь, что файл cardata.csv находится в рабочей директории Airflow")
        return False

def validate_data():
    """Проверка качества данных перед обучением"""
    try:
        df = pd.read_csv("cardata.csv")
        required_cols = ['Car_Name', 'Year', 'Selling_Price', 'Present_Price', 
                        'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
        
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"❌ Отсутствуют колонки: {missing}")
        
        print(f"✅ Валидация пройдена. Строк: {len(df)}, Колонки: {list(df.columns)}")
        return True
    except Exception as e:
        print(f"❌ Ошибка валидации: {e}")
        return False

def log_completion(**context):
    """Логирование завершения пайплайна"""
    ti = context['ti']
    task_results = {
        'download': ti.xcom_pull(task_ids='download_data'),
        'validate': ti.xcom_pull(task_ids='validate_data'),
        'train': ti.xcom_pull(task_ids='train_model'),
    }
    print(f"Результаты пайплайна: {task_results}")
    return task_results

# Конфигурация DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id="car_price_ml_pipeline",
    default_args=default_args,
    description="ML pipeline for car price prediction",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",  # ✅ Используем schedule вместо schedule_interval
    catchup=False,
    max_active_runs=1,
    tags=["ml", "car-price"],
)

# Задачи пайплайна
download_task = PythonOperator(
    task_id="download_data",
    python_callable=download_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id="validate_data",
    python_callable=validate_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id="train_model",
    python_callable=train,
    dag=dag,
)

log_task = PythonOperator(
    task_id="log_completion",
    python_callable=log_completion,
    trigger_rule='all_done',
    dag=dag,
)

# Зависимости
download_task >> validate_task >> train_task >> log_task
