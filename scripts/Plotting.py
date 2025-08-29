import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------
# VERİYİ YÜKLE
# --------------------------
df = pd.read_csv("./data/processed/all_cycles_metadata_core.csv")

os.makedirs("./grafikler", exist_ok=True)  # Grafiklerin kaydedileceği klasör

# --------------------------
# 1. Eksik Veri Isı Haritası
# --------------------------
plt.figure(figsize=(16, 6))
sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="YlOrRd")
plt.title("Veri Setinde Eksik Veri Isı Haritası", fontsize=14)
plt.xlabel("Özellikler (Sütunlar)")
plt.ylabel("Satırlar")
plt.tight_layout()
plt.savefig("./grafikler/eksik_veri_heatmap.png", dpi=300)
plt.close()

# --------------------------
# 2. Sütun Bazında Eksik Veri Oranları
# --------------------------
missing = df.isnull().mean().sort_values(ascending=False)
plt.figure(figsize=(12, 5))
missing.plot(kind="bar", color="tomato")
plt.title("Sütun Bazında Eksik Veri Oranı", fontsize=14)
plt.ylabel("Eksik Oran")
plt.xlabel("Özellikler")
plt.tight_layout()
plt.savefig("./grafikler/eksik_veri_oranlari.png", dpi=300)
plt.close()

# --------------------------
# 3. Batarya Bazında SoH'nin test_id'ye Göre Değişimi
# --------------------------
plt.figure(figsize=(10,6))
for batarya in df['battery_id'].unique():
    sub = df[(df['battery_id'] == batarya) & (~df['SoH_%'].isnull())]
    plt.plot(sub['test_id'], sub['SoH_%'], marker='o', label=batarya)
plt.title("Batarya Sağlık Durumu (SoH) - Döngü Bazında", fontsize=14)
plt.xlabel("Test Döngüsü (test_id)")
plt.ylabel("Sağlık Durumu (%)")
plt.legend(title="Batarya")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("./grafikler/SoH_vs_test_id.png", dpi=300)
plt.close()

# --------------------------
# 4. Batarya Bazında SoC'nin test_id'ye Göre Değişimi
# --------------------------
plt.figure(figsize=(10,6))
for batarya in df['battery_id'].unique():
    sub = df[(df['battery_id'] == batarya) & (~df['SoC_Progress_%'].isnull())]
    plt.plot(sub['test_id'], sub['SoC_Progress_%'], marker='o', label=batarya)
plt.title("Batarya Şarj Durumu (SoC) - Döngü Bazında", fontsize=14)
plt.xlabel("Test Döngüsü (test_id)")
plt.ylabel("Şarj Durumu (%)")
plt.legend(title="Batarya")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("./grafikler/SoC_vs_test_id.png", dpi=300)
plt.close()

# --------------------------
# 5. Korelasyon Matrisi ve Isı Haritası (SoH, SoC ve Temel Sayısal Özellikler)
# --------------------------
num_cols = [
    'Capacity', 'Current_measured_mean', 'Temperature_measured_mean', 
    'Voltage_measured_mean', 'SoH_%', 'SoC_Progress_%'
]
corr_df = df[num_cols].fillna(df[num_cols].mean())
corr = corr_df.corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Korelasyon Matrisi Isı Haritası (Eksik Değerler Ortalama ile Dolduruldu)", fontsize=14)
plt.tight_layout()
plt.savefig("grafikler/korelasyon_isiharitasi_dolgu.png", dpi=300)
plt.close()


# --------------------------
# 6. SoH ve Ortalama Sıcaklık İlişkisi (Her Batarya İçin Scatter)
# --------------------------
plt.figure(figsize=(9,6))
for batarya in df['battery_id'].unique():
    sub = df[(df['battery_id'] == batarya) & (~df['SoH_%'].isnull())]
    plt.scatter(sub['Temperature_measured_mean'], sub['SoH_%'], alpha=0.7, label=batarya)
plt.title("SoH vs Ortalama Sıcaklık", fontsize=14)
plt.xlabel("Ortalama Sıcaklık (°C)")
plt.ylabel("Sağlık Durumu (%)")
plt.legend(title="Batarya")
plt.tight_layout()
plt.savefig("./grafikler/SoH_vs_sicaklik.png", dpi=300)
plt.close()

# --------------------------
# 7. SoH ve Ortalama Akım İlişkisi (Her Batarya İçin Scatter)
# --------------------------
plt.figure(figsize=(9,6))
for batarya in df['battery_id'].unique():
    sub = df[(df['battery_id'] == batarya) & (~df['SoH_%'].isnull())]
    plt.scatter(sub['Current_measured_mean'], sub['SoH_%'], alpha=0.7, label=batarya)
plt.title("SoH vs Ortalama Akım", fontsize=14)
plt.xlabel("Ortalama Akım (A)")
plt.ylabel("Sağlık Durumu (%)")
plt.legend(title="Batarya")
plt.tight_layout()
plt.savefig("./grafikler/SoH_vs_akim.png", dpi=300)
plt.close()

print("TÜM GRAFİKLER 'grafikler/' KLASÖRÜNE KAYDEDİLDİ.")
