# nasa-battery-soh-soc-prediction-case
End-to-end SoH/SoC prediction on NASA battery data: feature extraction, EDA, ML/DL (XGBoost, LightGBM, MLP), model training, evaluation, model export, Dockerized FastAPI+Streamlit demo, reproducible pipeline. | NASA batarya verisiyle tam veri temizleme, görselleştirme, model eğitimi ve kaydı, Docker ile kolay dağıtım ve arayüz demo.


# NASA Batarya SoH/SoC Tahmin Pipeline'ı

Bu proje, NASA'nın batarya veri setiyle uçtan uca veri temizleme, feature çıkarımı, keşifsel veri analizi, makine öğrenmesi ve derin öğrenme tabanlı modelleme, API ve kullanıcı arayüzü ile SoH (State of Health) ve SoC (State of Charge) tahmini sağlar. Tüm pipeline, Docker ile kolayca dağıtılır ve yeniden üretilebilir.

---
```
## 🚀 Proje Yapısı

nasa-battery-soh-soc-prediction-case/
│
├── api/                     
│   └── fastAPI.py           # FastAPI backend (model servisi)
├── data/
│   ├── processed/           # Temizlenmiş & öznitelik çıkarılmış veri
│   └── raw/                 # Ham NASA batarya verisi
├── models/                  # Eğitilmiş modeller (.pkl, .pt)
├── scripts/
│   ├── Data_Clean.py        # Veri temizleme & öznitelik çıkarımı
│   ├── DL_Tabular_Regression.py # MLP tabanlı model eğitimi
│   ├── XGBoost_and_LightGBM.py  # ML modellerinin eğitimi ve kaydı
│   └── requirements.txt     # Scriptler için bağımlılıklar
├── ui/
│   └── streamlit_app.py     # Streamlit kullanıcı arayüzü
├── main.py                  # Pipeline ana giriş noktası
├── requirements.txt         # Docker & genel kullanım için bağımlılıklar
└── docker-compose.yml       # Docker orkestrasyonu

yaml
Copy code
```
---

🚀 Hızlı Başlangıç
1. Docker (Önerilen Yöntem)

Tüm pipeline’ı (API + UI) tek komutla başlatın:
```
docker-compose up --build
```

FastAPI servisi: localhost:8080

Streamlit arayüzü: localhost:8501

2. Manuel (Python ile)

Docker kullanılamıyorsa:
```
pip install -r requirements.txt
python scripts/Data_Clean.py
python scripts/XGBoost_and_LightGBM.py
python scripts/DL_Tabular_Regression.py
```
📊 Model Performansları
```
Hedef	         Model	   MAE	 RMSE	Eğitim Süresi	Tahmin Süresi	Model Dosyası
SoH_%	         XGBoost 	3.86	4.16	   0.37s	     0.013s	xgboost_model_SoH_%_discharge.pkl
SoH_%	         LightGBM	3.37	3.90	   0.13s	     0.003s	lightgbm_model_SoH_%_discharge.pkl
SoC_Progress_%	  XGBoost	 4.79	6.07	   0.36s	     0.010s	xgboost_model_SoC_Progress_%_discharge.pkl
SoC_Progress_%	LightGBM	4.97	6.09	   0.10s	     0.004s	lightgbm_model_SoC_Progress_%_discharge.pkl
SoH_%	         MLP (DL)	3.01	3.72	   5.03s	     0.000s	mlp_regressor_SoH_%_discharge.pt
SoC_Progress_%	MLP (DL)	5.49	7.49	   5.67s	     0.000s	mlp_regressor_SoC_Progress_%_discharge.pt
```
🎯 Temel Özellikler

Eksiksiz ML/DL Pipeline: Ham veri → öznitelik çıkarımı → EDA → model seçimi ve eğitimi → API → arayüz → docker

Model Karşılaştırmaları: SoH ve SoC regresyonunda XGBoost, LightGBM ve MLP (derin öğrenme) modelleri benchmark edilir

Keşifsel Veri Analizi: Korelasyon, eksik değerler, görselleştirme (bkz. /graphics)

Kullanıcı Arayüzü: Streamlit uygulaması ile toplu tahmin ve canlı demo

API-First: FastAPI mikroservis ile yüksek performanslı tahmin altyapısı

Tek Komutla Kurulum: Docker Compose ile tam otomasyon ve tekrarlanabilirlik

Modüler Mimari: Pipeline’ın her aşaması ayrı ve şeffaf

📦 Dağıtım

Üretime Hazır: Konteyner tabanlı, versiyonlu, kolay taşınabilir.

Genişletilebilir: Yeni ML/DL modelleri, ön işleme adımları veya veri kaynakları kolayca entegre edilebilir.

👨‍💻 Katkı

PR, issue ve önerilere açıktır.

MIT Lisansı ile lisanslanmıştır.

📄 Lisans

Bu depo MIT Lisansı
 ile lisanslanmıştır.

👤 Geliştirici

adnankerem
