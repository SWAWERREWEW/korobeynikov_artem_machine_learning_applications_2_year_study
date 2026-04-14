# train_with_mlflow.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import pickle
import os

# ------------------------------
# Функции предобработки (скопированы из вашего ноутбука)
# ------------------------------
def preprocessing_data_frame(frame):
    df = frame.copy()
    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    # ... (полностью ваш код из ipynb)
    # Здесь вставьте полную реализацию preprocessing_data_frame из вашего ноутбука
    # Для краткости я показываю только заглушку — вы замените на свой код.
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
# Основной пайплайн обучения
# ------------------------------
def train_and_log():
    # 1. Загрузка данных
    df = pd.read_csv('https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv')
    
    # 2. Предобработка
    df_proc = preprocessing_data_frame(df)
    X, y, scaler, power_trans = scale_frame(df_proc)
    
    # 3. Разбиение
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    y_train = y_train.ravel()
    y_val = y_val.ravel()
    
    # 4. Подбор гиперпараметров
    params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
              'l1_ratio': [0.001, 0.05, 0.01, 0.2]}
    base_lr = SGDRegressor(random_state=42)
    clf = GridSearchCV(base_lr, params, cv=5)
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    
    # 5. Предсказания (в исходной шкале цен)
    y_pred_scaled = best_model.predict(X_val)
    y_pred_orig = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_val_orig = power_trans.inverse_transform(y_val.reshape(-1, 1)).flatten()
    
    rmse, mae, r2 = eval_metrics(y_val_orig, y_pred_orig)
    
    # 6. Логирование в MLflow
    mlflow.set_tracking_uri("file:./mlruns")  # или ваш удалённый сервер
    with mlflow.start_run(run_name="training_from_script") as run:
        mlflow.log_params(best_model.get_params())
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Логируем обычную sklearn-модель (без обёртки)
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        # Также сохраняем scaler и power_transformer как артефакты
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        with open("power_trans.pkl", "wb") as f:
            pickle.dump(power_trans, f)
        mlflow.log_artifact("scaler.pkl")
        mlflow.log_artifact("power_trans.pkl")
        
        print(f"Run ID: {run.info.run_id}")
        print(f"R² = {r2:.4f}, RMSE = {rmse:.2f}, MAE = {mae:.2f}")
        
        return run.info.run_id

if __name__ == "__main__":
    train_and_log()
