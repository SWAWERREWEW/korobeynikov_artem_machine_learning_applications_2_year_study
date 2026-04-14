# log_wrapped_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import pickle
from model_wrapper import PricePredictor


mlflow.set_tracking_uri("file:./mlruns")  # можно оставить как есть

experiment_name = "car_price_experiment"
try:
    # Попробуем получить эксперимент по имени
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        # Создаём новый эксперимент
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Создан новый эксперимент с ID: {experiment_id}")
    else:
        experiment_id = exp.experiment_id
        print(f"Используем существующий эксперимент: {experiment_name} (ID: {experiment_id})")
    mlflow.set_experiment(experiment_name)
except Exception as e:
    print(f"Ошибка при работе с экспериментом: {e}")
    raise


def preprocessing_data_frame(frame):
    df = frame.copy()
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    num_columns = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Price(euro)']

    # Удаление выбросов (как в оригинале)
    question_dist = df[(df.Year < 2021) & (df.Distance < 1100)]
    df = df.drop(question_dist.index)

    question_dist = df[(df.Distance > 1e6)]
    df = df.drop(question_dist.index)

    question_engine = df[df["Engine_capacity(cm3)"] < 200]
    df = df.drop(question_engine.index)

    question_engine = df[df["Engine_capacity(cm3)"] > 5000]
    df = df.drop(question_engine.index)

    question_price = df[(df["Price(euro)"] < 101)]
    df = df.drop(question_price.index)

    question_price = df[df["Price(euro)"] > 1e5]
    df = df.drop(question_price.index)

    question_year = df[df.Year < 1971]
    df = df.drop(question_year.index)

    df = df.reset_index(drop=True)

    # OrdinalEncoder для категориальных признаков
    ordinal = OrdinalEncoder()
    ordinal.fit(df[cat_columns])
    Ordinal_encoded = ordinal.transform(df[cat_columns])
    df_ordinal = pd.DataFrame(Ordinal_encoded, columns=cat_columns)
    df[cat_columns] = df_ordinal[cat_columns]

    return df

def scale_frame(frame):
    df = frame.copy()
    X, y = df.drop(columns=['Price(euro)']), df['Price(euro)']
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X.values)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    return X_scaled, y_scaled, scaler, power_trans

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

# ------------------------------
# 2. Загрузка и подготовка данных
# ------------------------------
print("Загрузка данных...")
df_raw = pd.read_csv('https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv', delimiter=',')

print("Предобработка...")
df_proc = preprocessing_data_frame(df_raw)

print("Масштабирование...")
X, y, scaler, power_trans = scale_frame(df_proc)

# Преобразуем y в одномерный массив (для совместимости с SGDRegressor)
y = y.ravel()

# Разделение на обучающую и валидационную выборки
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ------------------------------
# 3. Подбор гиперпараметров и обучение
# ------------------------------
print("Подбор гиперпараметров (GridSearchCV)...")
params = {
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'l1_ratio': [0.001, 0.05, 0.01, 0.2]
}
base_model = SGDRegressor(random_state=42)
grid_search = GridSearchCV(base_model, params, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Лучшие параметры: alpha={best_model.alpha}, l1_ratio={best_model.l1_ratio}")

# Предсказание и оценка на валидации (в исходных ценах)
y_pred_scaled = best_model.predict(X_val)
y_pred_orig = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
y_val_orig = power_trans.inverse_transform(y_val.reshape(-1, 1)).flatten()

rmse, mae, r2 = eval_metrics(y_val_orig, y_pred_orig)
print(f"Валидационные метрики: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

# ------------------------------
# 4. Создание обёртки и логирование в MLflow
# ------------------------------
print("Логирование обёрнутой модели в MLflow...")
wrapped_model = PricePredictor(best_model, power_trans)

# Установите tracking URI (локальная папка или удалённый сервер)
mlflow.set_tracking_uri("file:./mlruns")   # артефакты будут в ./mlruns

with mlflow.start_run(run_name="wrapped_price_model") as run:
    # Логируем метрики и параметры
    mlflow.log_params(best_model.get_params())
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Логируем обёрнутую модель
    signature = infer_signature(X_train, wrapped_model.predict(None, X_train))
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapped_model,
        signature=signature,
        input_example=X_train[:5]
    )

    # Сохраняем scaler (если понадобится для новых данных)
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact("scaler.pkl")

    run_id = run.info.run_id
    print(f"\n✅ Модель успешно сохранена!")
    print(f"   Run ID: {run_id}")
    print(f"   Команда для serve: mlflow models serve -m runs:/{run_id}/model -p 5001 --env-manager local")
