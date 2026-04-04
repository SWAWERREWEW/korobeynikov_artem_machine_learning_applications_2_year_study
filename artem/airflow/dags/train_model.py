import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer, OrdinalEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import mlflow
import joblib
import os

def load_and_preprocess(filepath):
    """Загрузка и базовая очистка данных"""
    df = pd.read_csv(filepath)
    
    # Удаление пропусков
    df = df.dropna()
    
    # Удаление выбросов по логике предметной области
    df = df[(df['Kms_Driven'] > 0) & (df['Kms_Driven'] < 500000)]
    df = df[(df['Selling_Price'] > 0) & (df['Selling_Price'] < 100)]
    df = df[(df['Present_Price'] > 0) & (df['Present_Price'] < 100)]
    df = df[(df['Year'] >= 1990) & (df['Year'] <= 2025)]
    
    return df.reset_index(drop=True)

def encode_categorical(df, cat_columns):
    """Кодирование категориальных признаков"""
    df_encoded = df.copy()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df_encoded[cat_columns] = encoder.fit_transform(df[cat_columns])
    return df_encoded, encoder

def scale_features(X, y):
    """Масштабирование признаков и целевой переменной"""
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    
    X_scaled = scaler.fit_transform(X)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    return X_scaled, y_scaled, scaler, power_trans

def eval_metrics(actual, pred):
    """Расчёт метрик качества"""
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train():
    """Основная функция обучения модели"""
    # Пути к данным и артефактам
    data_path = "cardata.csv"
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # 1. Загрузка и предобработка
    df = load_and_preprocess(data_path)
    
    # Определение признаков и целевой переменной
    target = 'Selling_Price'
    cat_columns = ['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']
    num_columns = ['Year', 'Present_Price', 'Kms_Driven']
    
    # Кодирование категориальных признаков
    df_processed, encoder = encode_categorical(df, cat_columns)
    
    # Формирование X, y
    feature_cols = num_columns + cat_columns
    X = df_processed[feature_cols]
    y = df_processed[target]
    
    # Масштабирование
    X_scaled, y_scaled, scaler, power_trans = scale_features(X, y)
    
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
    )
    
    # Параметры для GridSearch
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.1, 0.2],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'fit_intercept': [True, False],
    }
    
    # Настройка MLflow
    mlflow.set_experiment("car_price_prediction")
    
    with mlflow.start_run():
        # 2. Обучение модели
        base_model = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
        grid_search = GridSearchCV(base_model, params, cv=3, n_jobs=4, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # 3. Валидация и метрики
        y_pred_scaled = best_model.predict(X_val)
        y_pred = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        y_val_original = power_trans.inverse_transform(y_val.reshape(-1, 1)).ravel()
        
        rmse, mae, r2 = eval_metrics(y_val_original, y_pred)
        
        # Логирование параметров и метрик
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # 4. Сохранение артефактов
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "model", signature=signature)
        
        # Сохранение локально для продакшена
        joblib.dump(best_model, os.path.join(artifacts_dir, "model.pkl"))
        joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))
        joblib.dump(power_trans, os.path.join(artifacts_dir, "power_transformer.pkl"))
        joblib.dump(encoder, os.path.join(artifacts_dir, "encoder.pkl"))
        
        print(f"✅ Обучение завершено! Метрики: RMSE={rmse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")
        print(f"📁 Артефакты сохранены в: {artifacts_dir}")
        
        return {"rmse": rmse, "mae": mae, "r2": r2}


if __name__ == "__main__":
    train()

