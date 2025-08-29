from scripts.Data_Clean import run_data_clean
from scripts.XGBoost_and_LightGBM import run_ml_pipeline
from scripts.DL_Tabular_Regression import run_dl_pipeline

def main():
    # --- Dosya/dizin ayarları ve pipeline parametreleri ---
    data_path = "./data/raw"
    output_folder_time_series = "./tmp"
    output_folder_single_value = "./data/processed"

    # Modelden hariç tutulacak sütunlar ve hedefler (drop_cols)
    drop_cols = [
        'uid', 'battery_id', 'test_id', 'filename', 'type', 'start_time', 'SoH_%', 'SoC_Progress_%'
    ]
    # Her iki model için de ana hedefler ve döngü tipi
    tasks = [("SoH_%", "discharge"), ("SoC_Progress_%", "discharge")]

    print("=== Starting Data Cleaning ===")
    # -- Veri temizleme ve öznitelik çıkarımı adımı --
    csv_path = run_data_clean(data_path, output_folder_time_series, output_folder_single_value)
    print(f"Data cleaned. Output: {csv_path}")

    print("=== Running ML Models ===")
    # -- ML (XGBoost & LightGBM) pipeline başlatılır --
    ml_results = run_ml_pipeline(csv_path, tasks, drop_cols)

    print("=== Running DL Model ===")
    # -- DL (PyTorch MLP) pipeline başlatılır --
    dl_results = run_dl_pipeline(csv_path, tasks, drop_cols, batch_size=32, n_epochs=250, lr=1e-3)

    print("\n======== FINAL RESULTS SUMMARY ========")
    print("ML Results:")
    # -- ML model sonuçları ekrana özet olarak yazdırılır --
    for (target, model_name), res in ml_results.items():
        print(f"{target} | {model_name} | {res['cycle_type']}: "
              f"MAE={res['mae']:.4f}, RMSE={res['rmse']:.4f}, "
              f"Train Time={res['train_time']:.2f}s, Predict Time={res['pred_time']:.3f}s, "
              f"Saved: {res['model_file']}")

    print("\nDL Results:")
    # -- DL model sonuçları ekrana özet olarak yazdırılır --
    for target, res in dl_results.items():
        print(f"{target} | {res['cycle_type']}: "
              f"MAE={res['mae']:.4f}, RMSE={res['rmse']:.4f}, "
              f"Train Time={res['train_time']:.2f}s, Predict Time={res['pred_time']:.3f}s, "
              f"Model: {res['model_file']}, Scaler: {res['scaler_file']}")

if __name__ == "__main__":
    # Komut satırından çağrıldığında ana pipeline başlatılır
    main()
