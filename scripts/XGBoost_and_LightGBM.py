import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os
import time

def run_ml_pipeline(input_csv, tasks, drop_cols, min_non_nan_ratio=0.5):
    """
    Tabular veri üzerinde regresyon için klasik ML pipeline (XGBoost & LightGBM).
    - Her hedef (ör: SoH, SoC Progress) ve cycle tipi için ayrı ayrı çalışır
    - Eksik veri oranı yüksek olmayan sayısal feature'lar otomatik seçilir
    - Median ile eksik doldurma yapılır
    - Train-test split batarya bazında rastgele
    - Hem XGBoost hem LightGBM ile ayrı modeller eğitilir
    - MAE/RMSE hesaplanır, model dosyaları kaydedilir
    - Tüm metrik ve dosya adları sonuç dict'ine eklenir
    """
    results = {}

    # Ana veri seti yüklenir
    df = pd.read_csv(input_csv)

    for target, relevant_type in tasks:
        # Sadece istenen döngü tipi (ör: discharge) ile ilerle
        df_sub = df[df["type"] == relevant_type].copy()
        if df_sub.empty:
            continue

        # --- Feature selection ---
        numeric_cols = [
            col for col in df_sub.columns
            if col not in drop_cols and df_sub[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        # %50'den fazla dolu olan sayısal sütunlar
        features = [col for col in numeric_cols if df_sub[col].notna().mean() > min_non_nan_ratio]

        # İlk kez çalıştırılıyorsa, feature listesi dışarı kaydedilir
        if not os.path.exists("feature_list.pkl"):
            joblib.dump(features, "feature_list.pkl")

        # Model girdisi ve hedefler
        X = df_sub[features]
        y = df_sub[target]

        # --- Eksik değerleri median ile doldur ---
        imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features, index=X.index)

        # --- Batarya bazında train-test split (veri sızıntısını önlemek için) ---
        unique_batteries = df_sub['battery_id'].unique()
        train_batteries, test_batteries = train_test_split(
            unique_batteries, test_size=0.33, random_state=42
        )
        train_mask = df_sub['battery_id'].isin(train_batteries)
        test_mask = df_sub['battery_id'].isin(test_batteries)
        X_train, X_test = X_imputed[train_mask], X_imputed[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # NaN veya Inf içeren sample'lar temizlenir
        train_nonan = (~np.isnan(y_train)) & (~np.isinf(y_train))
        test_nonan = (~np.isnan(y_test)) & (~np.isinf(y_test))
        X_train, y_train = X_train[train_nonan], y_train[train_nonan]
        X_test, y_test = X_test[test_nonan], y_test[test_nonan]

        # --- Model eğitim ve test ---
        for model_name, model in [
            ("xgboost_model", XGBRegressor(random_state=42, n_jobs=-1)),
            ("lightgbm_model", LGBMRegressor(random_state=42, n_jobs=-1))
        ]:
            # Eğitim zamanı ölçülür
            t0 = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - t0

            # Tahmin zamanı ve skorlar hesaplanır
            t0 = time.time()
            y_pred = model.predict(X_test)
            pred_time = time.time() - t0
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Model dosyası kaydı
            model_file = f"{model_name}_{target}_{relevant_type}.pkl"
            joblib.dump(model, model_file)

            # Sonuçları dict'e kaydet
            results[(target, model_name)] = {
                "cycle_type": relevant_type,
                "mae": mae,
                "rmse": rmse,
                "train_time": train_time,
                "pred_time": pred_time,
                "model_file": model_file
            }
    return results

if __name__ == "__main__":
    # Komut satırından çalıştırmak için örnek parametrelerle pipeline başlatılır
    run_ml_pipeline(
        "../tmp/all_cycles_metadata_core.csv",
        [("SoH_%", "discharge"), ("SoC_Progress_%", "discharge")],
        drop_cols=['uid', 'battery_id', 'test_id', 'filename', 'type', 'start_time', 'SoH_%', 'SoC_Progress_%']
    )
