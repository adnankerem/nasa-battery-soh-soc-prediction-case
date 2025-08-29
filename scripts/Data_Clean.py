import scipy.io
import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression

def load_files(directory):
    """
    Verilen klasördeki .mat dosyalarını listeler, okur ve içeriklerini Python nesnesine çevirir.
    Her dosya adı da ayrıca kaydedilir.
    """
    files = os.listdir(directory)
    data_list = []
    data_names = []
    for file in files:
        if file.endswith('.mat'):
            # .mat dosyasını yükle ve dosya ismini not et
            mat = scipy.io.loadmat(os.path.join(directory, file), simplify_cells=True)
            data_names.append(str(file.removesuffix('.mat')))
            data_list.append(mat)
    return data_list, data_names


def Data_Organization(cycle_data_dict):
    """
    Cycle içindeki veriyi ikiye ayırır:
    - time_series_data: Zaman serisi (ör. Voltage, Current...)
    - single_value_data: Her cycle için tekil değerler (Capacity, Re, Rct)
    """
    time_series_data = {}
    single_value_data = {}

    for key, value in cycle_data_dict.items():
        if key not in ['Re', 'Rct', 'Capacity']:
            time_series_data[key] = value
        else:
            single_value_data[key] = value
    return time_series_data, single_value_data


def extract_single_value_metadata(single_value_data):
    """
    Tekil veri sözlüğünden Capacity, Re ve Rct'yi çeker. 
    Eğer herhangi biri eksikse np.nan atanır.
    """
    capacity = single_value_data.get('Capacity', np.nan)
    re = single_value_data.get('Re', np.nan)
    rct = single_value_data.get('Rct', np.nan)
    return capacity, re, rct


def extract_selected_stats(series):
    """
    Bir zaman serisinden temel özet istatistikleri (mean, std, min, max, delta, slope) hesaplar.
    Delta: son değer - ilk değer
    Slope: Lineer regresyonla eğim (trend)
    Seride NaN varsa, slope sadece geçerli değerlerle hesaplanır.
    """
    n = len(series)
    x = np.arange(n).reshape(-1, 1)
    y = np.array(series).reshape(-1, 1)
    valid_mask = ~np.isnan(y).flatten()

    slope = np.nan
    if np.sum(valid_mask) > 1:
        try:
            # NaN olmayan noktalar ile eğim bulunur
            slope = LinearRegression().fit(x[valid_mask], y[valid_mask]).coef_[0, 0]
        except:
            pass

    return {
        'mean': np.nanmean(series),
        'std': np.nanstd(series),
        'min': np.nanmin(series),
        'max': np.nanmax(series),
        'delta': series[-1] - series[0] if len(series) > 1 else np.nan,
        'slope': slope
    }


def run_data_clean(data_path, output_folder_time_series, output_folder_single_value, data_name_string="all_cycles_metadata_core"):
    """
    Veri temizleme pipeline'ı burada:
    - Her .mat dosyasını döngü döngü gezerek okunur
    - Zaman serilerinden özet istatistikler hesaplanır, tekil değerler toplanır
    - Her döngü için satır bazlı metadata ve özetler birleştirilir
    - SoH (%) ve alternatif SoC Progress (%) eklenir
    - Sonuç CSV kaydedilir

    Dışarıdan argüman olarak klasörler ve çıktı dosya adı alınabilir.
    """
    os.makedirs(output_folder_time_series, exist_ok=True)

    # Ana kolonlar (test tipi, zaman, sıcaklık vs.)
    base_cols = [
        'type', 'start_time', 'ambient_temperature', 'battery_id', 'test_id',
        'uid', 'filename', 'Capacity', 'Re', 'Rct'
    ]
    # Temel sinyaller
    target_signals = ['Voltage', 'Current', 'Temperature', 'Time']

    dynamic_stat_cols = set()
    all_rows = []
    uid = 0

    # Tüm dosyalar yükleniyor
    data_list, data_names = load_files(data_path)

    for i in range(len(data_list)):
        battery_name = data_names[i]
        mat_data = data_list[i][battery_name]
        test_list = mat_data['cycle']

        for test_id, cycle in enumerate(test_list):
            uid += 1
            filename = str(uid).zfill(5) + ".csv"
            filepath = os.path.join(output_folder_time_series, filename)

            # Her cycle'dan zaman serisi ve tekil değerleri ayır
            time_series_data, single_value_data = Data_Organization(cycle['data'])

            # Sinyallerin uzunlukları farklıysa NaN ile eşitle
            maxlen = max(len(v) for v in time_series_data.values())
            for k, v in time_series_data.items():
                if len(v) < maxlen:
                    time_series_data[k] = list(v) + [np.nan] * (maxlen - len(v))

            # Zaman serisini CSV'ye kaydet
            time_series_df = pd.DataFrame.from_dict(time_series_data)
            time_series_df.to_csv(filepath, index=False)

            # Tekil değerleri çek (Capacity, Re, Rct)
            capacity, re, rct = extract_single_value_metadata(single_value_data)
            test_type = cycle.get('type', np.nan)
            test_start_time = cycle.get('time', np.nan)
            test_temperature = cycle.get('ambient_temperature', np.nan)

            # Ana satırı oluştur
            row = {
                'type': test_type,
                'start_time': test_start_time,
                'ambient_temperature': test_temperature,
                'battery_id': battery_name,
                'test_id': test_id,
                'uid': uid,
                'filename': filename,
                'Capacity': capacity,
                'Re': re,
                'Rct': rct
            }

            # Her ana sinyal için özet istatistik ekle (mean, std, ... slope vs.)
            for sig in target_signals:
                col_match = [c for c in time_series_df.columns if sig.lower() in c.lower()]
                if not col_match:
                    continue
                col = col_match[0]
                stats = extract_selected_stats(time_series_df[col].values)
                for stat_name, value in stats.items():
                    feat_name = f"{col}_{stat_name}"
                    row[feat_name] = value
                    dynamic_stat_cols.add(feat_name)

            all_rows.append(row)

    # Tüm cycle'lar tek bir tabloya yazılır
    metadata_cols = base_cols + sorted(dynamic_stat_cols)
    df_out = pd.DataFrame(all_rows, columns=metadata_cols)

    # ---- SoH ve SoC_Progress Hesaplama ----

    # SoH (%): Her battery'nin ilk kapasitesine göre normalize edilir
    df_out['SoH_%'] = np.nan
    for battery in df_out['battery_id'].unique():
        idx = df_out['battery_id'] == battery
        cap_series = df_out.loc[idx, 'Capacity'].dropna()
        if cap_series.empty:
            continue
        initial_capacity = cap_series.iloc[0]
        df_out.loc[idx, 'SoH_%'] = df_out.loc[idx, 'Capacity'] / initial_capacity * 100

    # Tabloda sıralama yapılır (daha düzenli çıktı için)
    df_out = df_out.sort_values(['battery_id', 'test_id'])

    # SoC Progress (%): Discharge döngülerindeki kalan kapasitenin kümülatif oranı
    df_out['SoC_Progress_%'] = np.nan
    for battery in df_out['battery_id'].unique():
        mask = (df_out['battery_id'] == battery) & (df_out['type'] == 'discharge')
        sub = df_out.loc[mask, 'Capacity']
        total_capacity = sub.sum()
        # Discharge döngüleri tersten kümülatiflenir, yüzdeye çevrilir
        reversed_sum = np.cumsum(sub[::-1])[::-1]
        df_out.loc[mask, 'SoC_Progress_%'] = reversed_sum / total_capacity * 100

    # Nihai tablo dosyası olarak kaydedilir
    out_path = os.path.join(output_folder_single_value, data_name_string + ".csv")
    df_out.to_csv(out_path, index=False)

    return out_path


if __name__ == "__main__":
    # Komut satırından çağrıldığında default argümanlarla çalıştırılır
    run_data_clean("../data", "../tmp")
