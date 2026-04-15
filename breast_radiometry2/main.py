import numpy as np
import matplotlib
#matplotlib.use('Agg')  # ⚡ Должно быть СТРОГО ДО импорта pyplot
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, distance_transform_edt
import time

from model import BreastRadiometryModelReal
from visualization import (
    plot_main_results, plot_tissue_composition, plot_breast_anatomy,
    plot_temperature_gradient, plot_temperature_contours, plot_temperature_difference_map,
    plot_sensitivity_kernels, plot_measurement_data, plot_temperature_histogram,
    plot_cross_section, plot_emissivity_map
)
from statistics import print_full_statistics

if __name__ == "__main__":
    try: plt.style.use('seaborn-v0_8')
    except: plt.style.use('default')
    
    print("="*70)
    print("🔬 МОДЕЛЬ РАДИОМЕТРИИ МОЛОЧНОЙ ЖЕЛЕЗЫ (ВЫСОКОЕ РАЗРЕШЕНИЕ)")
    print("="*70)

    RESOLUTION_PRESETS = {
        'low':    {'shape': (80, 100),  'tumor_radius': 8,  'resolution_mm': 4},
        'medium': {'shape': (160, 200), 'tumor_radius': 12, 'resolution_mm': 2},
        'high':   {'shape': (320, 400), 'tumor_radius': 16, 'resolution_mm': 1},
        'ultra':  {'shape': (480, 600), 'tumor_radius': 20, 'resolution_mm': 0.5}
    }

    quality = 'medium'
    preset = RESOLUTION_PRESETS[quality]
    print(f"\n📐 Режим: {quality.upper()} ({preset['shape'][0]}×{preset['shape'][1]} пикселей)")

    birads = 'B'
    model = BreastRadiometryModelReal(freq_ghz=3.0, resolution_mm=preset['resolution_mm'], birads_category=birads)

    print("\n📌 Генерация анатомического фантома...")
    start_total = time.time()
    tumor_position = (int(preset['shape'][0] * 0.56), int(preset['shape'][1] * 0.5))

    eps_map, cond_map, temp_true, breast_mask, areola_mask, nipple_mask, body_mask, tissue_type_map = model.create_anatomical_phantom(
        shape=preset['shape'], tumor_radius=preset['tumor_radius'], tumor_pos=tumor_position
    )

    h, w = temp_true.shape

    # 🔥 Найти реальную границу груди
    breast_rows = np.where(breast_mask.any(axis=1))[0]
    
    if len(breast_rows) > 0:
        breast_y_top = int(breast_rows.min())
        breast_y_bottom = int(breast_rows.max())
        print(f"📏 Границы груди: Y={breast_y_top}-{breast_y_bottom}")
        
        # 🔥 Сканировать на 30% глубины от верха груди
        scan_y = int(breast_y_top + (breast_y_bottom - breast_y_top) * 0.30)
        print(f"📍 Позиция сканера: Y={scan_y} (оптимальная глубина)")
    else:
        # Фоллбэк, если маска по какой-то причине пуста
        scan_y = int(h * 0.35)
        print(f"⚠️ Границы не найдены, используется резервная позиция Y={scan_y}")
        
    # 🔥 Формирование сетки антенн
    x_pos = np.linspace(int(w * 0.20), int(w * 0.80), 25, dtype=int)
    scan_grid = [(scan_y, x) for x in x_pos]
    
    print(f"📡 Количество антенн: {len(scan_grid)}")
    print(f"📍 Позиция сканера: Y={scan_y}")

    print("\n📡 Выполнение прямого сканирования...")
    Tb_data, emissivity_avg = model.forward_scan(eps_map, cond_map, temp_true, breast_mask, scan_grid)

    print("\n🔍 Диагностика Tb (в Кельвинах):")
    print(f"   Мин Tb: {Tb_data.min():.2f} K ({Tb_data.min() - 273.15:.2f}°C)\n   Макс Tb: {Tb_data.max():.2f} K ({Tb_data.max() - 273.15:.2f}°C)\n   Среднее Tb: {Tb_data.mean():.2f} K ({Tb_data.mean() - 273.15:.2f}°C)\n   Emissivity: {emissivity_avg.mean():.3f} ± {emissivity_avg.std():.3f}")
    if Tb_data.min() < 260 or Tb_data.max() > 320:
        print("   ❌ ВНИМАНИЕ: Tb вне физиологического диапазона (260-320 K)!\n   ⚠️ Проверьте конвертацию °C → K в forward_scan()!")
    else: print("   ✅ Tb в физиологическом диапазоне (260-320 K)")
    print("   ✅ Emissivity в норме (0.90-0.99)" if emissivity_avg.mean() >= 0.90 else "   ⚠️ Emissivity слишком низкий!")
    print("   ✅ Emissivity имеет вариацию" if emissivity_avg.std() >= 0.01 else "   ⚠️ Emissivity без вариации!")

    print("\n🔄 Реконструкция температуры...")
    Tb_noisy = Tb_data + np.random.normal(0, 0.05, size=Tb_data.shape)
    temp_recon = model.reconstruct_simple(Tb_noisy, emissivity_avg, scan_grid, temp_true.shape, breast_mask)

    print("\n📊 Генерация графиков...")
    plot_main_results(temp_true, temp_recon, breast_mask, model.tumor_center, areola_mask, nipple_mask, body_mask)
    plot_tissue_composition(tissue_type_map, breast_mask, areola_mask, nipple_mask, body_mask, birads)
    plot_breast_anatomy(eps_map, breast_mask, areola_mask, nipple_mask, body_mask)
    plot_temperature_gradient(temp_true, breast_mask, model.tumor_center)
    plot_temperature_contours(temp_true, breast_mask, model.tumor_center)
    plot_temperature_difference_map(temp_true, tissue_type_map, breast_mask)
    plot_sensitivity_kernels(model, breast_mask, scan_grid, n_show=5)
    plot_measurement_data(Tb_data, Tb_noisy, emissivity_avg, scan_grid)
    plot_temperature_histogram(temp_true, temp_recon, breast_mask)
    plot_cross_section(temp_true, temp_recon, breast_mask, model.tumor_center)
    plot_emissivity_map(eps_map, breast_mask)

    print_full_statistics(temp_true, temp_recon, breast_mask, Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, model, tissue_type_map)

    total_elapsed = time.time() - start_total
    print(f"\n⏱️ Общее время выполнения: {total_elapsed:.2f} сек ({total_elapsed/60:.1f} мин)")
    print("\n✅ Все графики сохранены в файлы 01_*.png ... 11_*.png")
    print("="*70)