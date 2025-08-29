# nasa-battery-soh-soc-prediction-case
nd-to-end SoH/SoC prediction on NASA battery data: feature extraction, EDA, ML/DL (XGBoost, LightGBM, MLP), model training, evaluation, model export, Dockerized FastAPI+Streamlit demo, reproducible pipeline. | NASA batarya verisiyle tam veri temizleme, görselleştirme, model eğitimi ve kaydı, Docker ile kolay dağıtım ve arayüz demo.


# NASA Batarya SoH/SoC Tahmin Pipeline'ı

Bu proje, NASA'nın batarya veri setiyle uçtan uca veri temizleme, öznitelik çıkarımı, keşifsel veri analizi, makine öğrenmesi ve derin öğrenme tabanlı modelleme, API ve kullanıcı arayüzü ile SoH (State of Health) ve SoC (State of Charge) tahmini sağlar. Tüm pipeline, Docker ile kolayca dağıtılır ve yeniden üretilebilir.

---

## 🚀 Proje Yapısı

api/
fastAPI.py # FastAPI backend (model servisi)
data/
processed/ # Temizlenmiş ve öznitelik çıkarılmış veriler
raw/ # Ham NASA batarya verisi
models/ # Eğitilmiş model ve scaler dosyaları (.pkl, .pt)
scripts/
Data_Clean.py # Veri temizleme & öznitelik çıkarımı
DL_Tabular_Regression.py # MLP tabanlı DL model eğitimi
XGBoost_and_LightGBM.py # ML model eğitimi ve kayıt
ui/
streamlit_app.py # Streamlit arayüzü
main.py # Pipeline yönetimi
requirements.txt # Gerekli Python paketleri
docker-compose.yml # Docker orkestrasyonu

yaml
Copy code

---

## 📦 Kurulum ve Çalıştırma

**Docker ile (Tavsiye Edilen):**
```bash
docker-compose up --build
FastAPI servisi (8080) ve Streamlit arayüzü (8501) başlar.

Manuel (Python ile):

bash
Copy code
pip install -r requirements.txt
python scripts/Data_Clean.py
python scripts/XGBoost_and_LightGBM.py
python scripts/DL_Tabular_Regression.py
📊 Model Sonuçları
Hedef	Model	MAE	RMSE	Eğitim Süresi	Tahmin Süresi	Model Dosyası
SoH_%	XGBoost	3.86	4.16	0.37s	0.013s	xgboost_model_SoH_%_discharge.pkl
SoH_%	LightGBM	3.37	3.90	0.13s	0.003s	lightgbm_model_SoH_%_discharge.pkl
SoC_Progress_%	XGBoost	4.79	6.07	0.36s	0.010s	xgboost_model_SoC_Progress_%_discharge.pkl
SoC_Progress_%	LightGBM	4.97	6.09	0.10s	0.004s	lightgbm_model_SoC_Progress_%_discharge.pkl
SoH_%	MLP (DL)	3.01	3.72	5.03s	0.000s	mlp_regressor_SoH_%_discharge.pt
SoC_Progress_%	MLP (DL)	5.49	7.49	5.67s	0.000s	mlp_regressor_SoC_Progress_%_discharge.pt

🎯 Temel Özellikler
Ham veri → öznitelik çıkarımı → keşifsel analiz (EDA) → model seçimi → eğitim → API → arayüz → docker

ML ve DL model karşılaştırması (XGBoost, LightGBM, MLPRegressor)

Eksik veri ve korelasyon analizi, grafikler (grafikler/ klasörü)

Streamlit ile toplu tahmin ve kullanıcı arayüzü

Docker ile kolay kurulum ve dağıtım

Lisans ve Katkı
MIT Lisansı ile lisanslanmıştır.
Katkıda bulunmak ve öneriler için PR veya issue açabilirsiniz.
Repo sahibi: adnankerem
