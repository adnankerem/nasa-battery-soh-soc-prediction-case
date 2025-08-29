import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import time

# --- Basit bir tabular MLP regresyon modeli tanımı ---
class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # 2 katmanlı bir fully connected ağ + ReLU aktivasyonu
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Girdi için tahmin çıktısı üretir
        return self.net(x)


def run_dl_pipeline(input_csv, tasks, drop_cols, batch_size=32, n_epochs=250, lr=1e-3):
    """
    Tabular veri ile temel regresyon için PyTorch tabanlı MLP pipeline'ı.
    Her hedef değişken (SoH, SoC Progress) için:
    - Gerekli veriler filtrelenir ve feature seçimi otomatik yapılır
    - Eksik değerler median ile doldurulur
    - Train-test ayrımı batarya bazında, rastgele yapılır
    - Veriler scale edilir (StandardScaler)
    - PyTorch tensöre çevrilip model eğitilir
    - MAE/RMSE ile test sonuçları hesaplanır, model ve scaler kaydedilir
    - Tüm önemli metrikler ve dosya isimleri sonuçlar dict'ine eklenir
    """
    results = {}
    df = pd.read_csv(input_csv)

    for TARGET, relevant_type in tasks:
        # Sadece istenen cycle tipi (ör: discharge) seçiliyor
        df_sub = df[df["type"] == relevant_type].copy()
        if df_sub.empty:
            continue

        # --- Feature selection ---
        numeric_cols = [
            col for col in df_sub.columns
            if col not in drop_cols and df_sub[col].dtype in [np.float64, np.float32, np.int64, np.int32]
        ]
        # Yeterli doluluk oranı (>%50) olan sütunlar alınır
        features = [col for col in numeric_cols if df_sub[col].notna().mean() > 0.5]

        X = df_sub[features]
        y = df_sub[TARGET]

        # --- Eksik değerler için median ile doldurma ---
        imp = SimpleImputer(strategy='median')
        X = pd.DataFrame(imp.fit_transform(X), columns=features, index=X.index)

        # --- Batarya bazında train-test split (veri sızıntısını engellemek için) ---
        unique_batteries = df_sub['battery_id'].unique()
        train_batteries, test_batteries = train_test_split(unique_batteries, test_size=0.33, random_state=42)
        train_mask = df_sub['battery_id'].isin(train_batteries)
        test_mask = df_sub['battery_id'].isin(test_batteries)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # --- NaN ve Inf değerleri temizle ---
        train_nonan = (~np.isnan(y_train)) & (~np.isinf(y_train))
        test_nonan = (~np.isnan(y_test)) & (~np.isinf(y_test))
        X_train, y_train = X_train[train_nonan], y_train[train_nonan]
        X_test, y_test = X_test[test_nonan], y_test[test_nonan]

        # --- Verilerin ölçeklenmesi ---
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- PyTorch tensöre dönüştürme ---
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # --- PyTorch DataLoader ---
        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # --- Model kurulumu ve eğitim ---
        model = MLP(input_dim=X_train.shape[1])
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        t0 = time.time()
        for epoch in range(n_epochs):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()
        train_time = time.time() - t0

        # --- Test aşaması ve skor hesaplama ---
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            y_pred = model(X_test_tensor).cpu().numpy().flatten()
        pred_time = time.time() - t0
        y_true = y_test_tensor.cpu().numpy().flatten()
        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_pred - y_true) ** 2))

        # --- Model ve scaler kaydı ---
        model_file = f"./models/mlp_regressor_{TARGET}_{relevant_type}.pt"
        scaler_file = f"./models/scaler_{TARGET}_{relevant_type}.pkl"
        torch.save(model.state_dict(), model_file)
        joblib.dump(scaler, scaler_file)

        # --- Sonuçların dict'e eklenmesi ---
        results[TARGET] = {
            "cycle_type": relevant_type,
            "mae": mae,
            "rmse": rmse,
            "train_time": train_time,
            "pred_time": pred_time,
            "model_file": model_file,
            "scaler_file": scaler_file
        }

    return results


if __name__ == "__main__":
    # Örnek kullanım için, discharge döngüsü ve ana hedefler üzerinde pipeline çalıştırılır
    run_dl_pipeline(
        "./tmp/all_cycles_metadata_core.csv",
        [("SoH_%", "discharge"), ("SoC_Progress_%", "discharge")],
        drop_cols=['uid', 'battery_id', 'test_id', 'filename', 'type', 'start_time', 'SoH_%', 'SoC_Progress_%']
    )
