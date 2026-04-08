from datetime import datetime
import json
import logging
import os
import pandas as pd

from airflow.sdk import DAG
from airflow.providers.standard.operators.python import PythonOperator
from hooks import CarsHook

logger = logging.getLogger(__name__)

def _fetch_cars(conn_id: str, templates_dict: dict, batch_size: int = 1000, **_):
    output_path = templates_dict["output_path"]
    logger.info("Fetching all cars from the API...")
    hook = CarsHook(conn_id=conn_id)
    cars = list(hook.get_cars(batch_size=batch_size))
    logger.info(f"Fetched {len(cars)} car records")

    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(cars, f)
    logger.info(f"Saved raw cars to {output_path}")

def _clean_cars_data(templates_dict: dict, **_):
    input_path = templates_dict["input_path"]
    output_path = templates_dict["output_path"]
    
    logger.info(f"Loading raw data from {input_path}")
    with open(input_path, "r") as f:
        cars_raw = json.load(f)
    
    df = pd.DataFrame(cars_raw)
    logger.info(f"Loaded {len(df)} records")

    # 1. Удаление дубликатов (по всем колонкам)
    before_dedup = len(df)
    df = df.drop_duplicates()
    logger.info(f"Removed {before_dedup - len(df)} duplicate rows")

    # 2. Удаление строк с пропусками (если важны все поля)
    before_dropna = len(df)
    df = df.dropna()
    logger.info(f"Removed {before_dropna - len(df)} rows with missing values")

    # 3. Преобразование категориальных признаков в числовые
    #    Словари маппинга (можно расширить при появлении новых значений)
    fuel_mapping = {
        "Petrol": 0,
        "Diesel": 1,
        "Hybrid": 2,
        "Electric": 3,
        "Metan/Propan": 4,
        "LPG": 5,
        "Ethanol": 6
    }
    transmission_mapping = {
        "Manual": 0,
        "Automatic": 1,
        "Manul": 0,        # опечатка в данных, исправляем на Manual -> 0
        "Tiptronic": 1,
        "CVT": 1
    }
    
    df["Fuel_type_code"] = df["Fuel_type"].map(fuel_mapping).fillna(-1).astype(int)
    df["Transmission_code"] = df["Transmission"].map(transmission_mapping).fillna(-1).astype(int)
    
    logger.info("Added numerical columns: Fuel_type_code, Transmission_code")
    
    # 4. Сохранение очищенного датасета
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    df.to_json(output_path, orient="records", indent=2)
    logger.info(f"Saved cleaned data to {output_path}")

with DAG(
    dag_id="02_hook",
    description="Fetches car data using CarsHook and cleans it",
    start_date=datetime(2026, 2, 3),
    schedule="@daily",
    catchup=False,
    max_active_runs=1,
) as dag:
    
    fetch = PythonOperator(
        task_id="fetch_cars",
        python_callable=_fetch_cars,
        op_kwargs={"conn_id": "carsapi"},
        templates_dict={"output_path": "/data/custom_hook/cars.json"},
    )
    
    clean = PythonOperator(
        task_id="clean_cars_data",
        python_callable=_clean_cars_data,
        templates_dict={
            "input_path": "/data/custom_hook/cars.json",
            "output_path": "/data/cleaned/cars_cleaned.json"
        },
    )
    
    fetch >> clean
