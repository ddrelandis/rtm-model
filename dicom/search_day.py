import os
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import pydicom
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ============================================================
# 1. УТИЛИТЫ ДЛЯ РАБОТЫ С DICOM
# ============================================================

def safe_get(ds, tag_name, default=None):
    """Безопасное получение тега DICOM по имени."""
    try:
        return getattr(ds, tag_name, default)
    except Exception:
        return default


def parse_dicom_date(da_value):
    """Парсит дату DICOM (формат DA: YYYYMMDD) в datetime.date."""
    if da_value is None:
        return None
    try:
        # pydicom >= 2.0 возвращает DA-объект, преобразуем в строку
        return datetime.strptime(str(da_value), "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


def extract_study_info(dicom_path):
    """Извлекает ключевую информацию из одного DICOM-файла."""
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
    except Exception as e:
        print(f"[!] Ошибка чтения {dicom_path}: {e}")
        return None

    return {
        "file": str(dicom_path),
        "patient_id": safe_get(ds, "PatientID", "UNKNOWN"),
        "study_instance_uid": safe_get(ds, "StudyInstanceUID"),
        "series_instance_uid": safe_get(ds, "SeriesInstanceUID"),
        "modality": safe_get(ds, "Modality"),
        "study_description": safe_get(ds, "StudyDescription"),
        "series_description": safe_get(ds, "SeriesDescription"),
        "study_date": parse_dicom_date(safe_get(ds, "StudyDate")),
        "series_date": parse_dicom_date(safe_get(ds, "SeriesDate")),
        "acquisition_date": parse_dicom_date(safe_get(ds, "AcquisitionDate")),
        # Ключевые теги для восстановления временной шкалы
        "temporal_offset_days": safe_get(ds, "LongitudinalTemporalOffsetFromEvent"),  # (0012,0052)
        "event_type": safe_get(ds, "LongitudinalTemporalEventType"),                  # (0012,0053)
    }


# ============================================================
# 2. СКАНИРОВАНИЕ ПАПКИ С DICOM
# ============================================================

def scan_dicom_folder(root_dir):
    """Рекурсивно сканирует папку и извлекает метаданные из всех DICOM-файлов."""
    root = Path(root_dir)
    records = []
    dicom_files = list(root.rglob("*.dcm")) + list(root.rglob("*.DCM"))

    # Также ищем файлы без расширения (типично для TCIA)
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() not in (".dcm", ".json", ".xml"):
            dicom_files.append(f)

    print(f"[i] Найдено файлов для анализа: {len(dicom_files)}")
    for f in dicom_files:
        info = extract_study_info(f)
        if info and info["study_instance_uid"]:
            records.append(info)

    return pd.DataFrame(records)


# ============================================================
# 3. ПОСТРОЕНИЕ ВРЕМЕННОЙ ШКАЛЫ ПАЦИЕНТА
# ============================================================

BASE_DATE = datetime(1960, 1, 1).date()  # Базовая дата TCIA

def build_patient_timeline(df, patient_id):
    patient_df = df[df["patient_id"] == patient_id].copy()
    if patient_df.empty:
        raise ValueError(f"Пациент {patient_id} не найден")

    # Находим самую раннюю дату для этого пациента (его реальный "якорь", например, 17.04.1959)
    # Игнорируем NaT (Not a Time), если где-то даты не распарсились
    valid_dates = patient_df['study_date'].dropna()
    if valid_dates.empty:
        return pd.DataFrame()
        
    anchor_date = valid_dates.min()
    
    studies = []
    for study_uid, group in patient_df.groupby("study_instance_uid"):
        first_row = group.iloc[0]
        study_date = first_row["study_date"]
        
        # Вычисляем смещение в днях ОТНОСИТЕЛЬНО РЕАЛЬНОГО ЯКОРЯ ПАЦИЕНТА
        offset_days = (study_date - anchor_date).days if pd.notnull(study_date) else None
        
        studies.append({
            "study_uid": study_uid,
            "modality": first_row["modality"],
            "description": first_row["study_description"],
            "absolute_date": study_date, # То, что написано в DICOM (напр. 17.04.1959)
            "days_from_baseline": offset_days, # Главный параметр для 3D-моделирования (0, 3, 45...)
            "num_series": len(group),
        })

    return pd.DataFrame(studies).sort_values("absolute_date")
# ============================================================
# 4. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_timeline(timeline_df, patient_id, save_path=None):
    """Визуализирует временную шкалу исследований пациента."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Цветовое кодирование по модальности
    modalities = timeline_df["modality"].unique()
    color_map = {m: plt.cm.tab10(i / max(1, len(modalities) - 1)) for i, m in enumerate(modalities)}

    for _, row in timeline_df.iterrows():
        date = row["reconstructed_date"]
        color = color_map.get(row["modality"], "gray")
        ax.scatter(date, 0, s=200, color=color, zorder=3, edgecolors="black")

        # Подпись: модальность + смещение в днях
        label = f"{row['modality']}"
        if row["offset_from_registration_days"] is not None:
            label += f"\n(+{row['offset_from_registration_days']:.0f}d)"
        ax.annotate(
            label,
            (date, 0),
            textcoords="offset points",
            xytext=(0, 15),
            ha="center",
            fontsize=9,
        )

    # Базовая линия
    ax.axhline(0, color="gray", linewidth=1, linestyle="--")
    ax.axvline(BASE_DATE, color="red", linestyle=":", label="Дата регистрации (01.01.1960)")

    ax.set_yticks([])
    ax.set_title(f"Временная шкала пациента: {patient_id}", fontsize=14)
    ax.set_xlabel("Дата (относительно регистрации)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%Y"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.xticks(rotation=45)
    ax.legend(loc="upper right")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[+] График сохранен: {save_path}")
    plt.show()


# ============================================================
# 5. ТОЧКА ВХОДА
# ============================================================

def main():
    # === НАСТРОЙКИ ===
    DICOM_ROOT = r"C:\data\cmb-brca\patient_001"  # Путь к папке с DICOM пациента
    PATIENT_ID = None  # Если None — возьмем первого найденного
    OUTPUT_CSV = "patient_timeline.csv"
    OUTPUT_PLOT = "patient_timeline.png"

    # === ШАГ 1: Сканирование ===
    print("[1/4] Сканирование DICOM-папки...")
    df = scan_dicom_folder(DICOM_ROOT)
    if df.empty:
        print("[!] Не найдено ни одного валидного DICOM-файла.")
        return

    print(f"[+] Всего записей: {len(df)}")
    print(f"[+] Найдено пациентов: {df['patient_id'].nunique()}")
    print(f"    ID: {df['patient_id'].unique().tolist()}")

    # === ШАГ 2: Выбор пациента ===
    patient_id = PATIENT_ID or df["patient_id"].iloc[0]
    print(f"\n[2/4] Построение шкалы для пациента: {patient_id}")

    # === ШАГ 3: Построение шкалы ===
    timeline_df = build_patient_timeline(df, patient_id)
    print(timeline_df.to_string(index=False))

    # === ШАГ 4: Сохранение и визуализация ===
    print(f"\n[3/4] Сохранение в {OUTPUT_CSV}")
    timeline_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"[4/4] Визуализация...")
    plot_timeline(timeline_df, patient_id, save_path=OUTPUT_PLOT)


if __name__ == "__main__":
    main()