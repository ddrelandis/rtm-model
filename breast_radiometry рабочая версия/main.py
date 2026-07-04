import numpy as np
import matplotlib
#matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, distance_transform_edt
import time

from model import BreastRadiometryModelReal
from visualization import (
    plot_main_results, plot_tissue_composition, plot_breast_anatomy,
    plot_temperature_gradient, plot_temperature_contours, plot_temperature_difference_map,
    plot_sensitivity_kernels, plot_measurement_data, plot_temperature_histogram,
    plot_cross_section, plot_emissivity_map, plot_antenna_coverage
)
from statistics import print_full_statistics

if __name__ == "__main__":
    try: plt.style.use('seaborn-v0_8')
    except: plt.style.use('default')
    
    RESOLUTION_PRESETS = {
        'low':    {'shape': (80, 100),  'tumor_radius': 8,  'resolution_mm': 4},
        'medium': {'shape': (160, 200), 'tumor_radius': 12, 'resolution_mm': 2},
        'high':   {'shape': (320, 400), 'tumor_radius': 16, 'resolution_mm': 1},
        'ultra':  {'shape': (480, 600), 'tumor_radius': 20, 'resolution_mm': 0.5}
    }
    
    c_scan = 0.4       #расстояние от антенны до мж
    num_ant = 40        #кол-во антенн
    quality = 'high'
    preset = RESOLUTION_PRESETS[quality]
    print(f"\nРежим: {quality.upper()} ({preset['shape'][0]}×{preset['shape'][1]} пикселей)")

    #плотность железистой ткани. Влияет на диэлектрические свойства и анатомию.
    birads = 'B'
    model = BreastRadiometryModelReal(freq_ghz=3.0, resolution_mm=preset['resolution_mm'], birads_category=birads)

    print("\nГенерация анатомического фантома...")
    start_total = time.time()
    #расположение опухоли
    tumor_position = (int(preset['shape'][0] * 0.56), int(preset['shape'][1] * 0.5))

    eps_map, cond_map, temp_true, breast_mask, areola_mask, nipple_mask, body_mask, tissue_type_map = model.create_anatomical_phantom(
        shape=preset['shape'], tumor_radius=preset['tumor_radius'], tumor_pos=tumor_position
    )

    h, w = temp_true.shape

    # поиск границы МЖ по оси Y
    breast_rows = np.where(breast_mask.any(axis=1))[0]
    
    if len(breast_rows) > 0:
        breast_y_top = int(breast_rows.min())
        breast_y_bottom = int(breast_rows.max())
        print(f"Границы молочной железы: Y={breast_y_top}-{breast_y_bottom}")
        
        # сканировать на 30% глубины от верха мж
        scan_y = int(breast_y_top + (breast_y_bottom - breast_y_top) * c_scan)
        print(f"Позиция сканера: Y={scan_y}")
    else:
        # если маска по какой-то причине пуста
        scan_y = int(h * 0.35)
        print(f"Границы не найдены, используется резервная позиция Y={scan_y}")
        
    # формирование сетки антенн
 
    x_pos = np.linspace(int(w * 0.20), int(w * 0.80), num_ant, dtype=int)
    scan_grid = [(scan_y, x) for x in x_pos]
    
    # ============================================================
    # 📡 РУЧНОЕ ЗАДАНИЕ ПОЗИЦИЙ АНТЕНН
    # ============================================================
    # Формат: список кортежей (y, x). 
    # y — строка (глубина), x — столбец (ширина)
    # Антенны должны находиться внутри breast_mask (на ткани или у границы)
    '''
    manual_antenna_positions = [
        # Верхняя центральная часть (над соском)
        (25, 200), (20, 200), 
        
        # Верхняя дуга (ближе к центру)
        (30, 170), (30, 230),
        (40, 140), (40, 260),
        (50, 120), (50, 280),
        
        # Средняя дуга (по бокам, снаружи)
        (65, 100), (65, 300),
        (80, 80), (80, 320),
        (100, 60), (100, 340),
        
        # Нижняя дуга (далеко по бокам)
        (120, 40), (120, 360),
        (150, 20), (150, 380),
        (170, 5), (170, 395),
        
        # Дополнительные точки для плотности
        (30, 100), (30, 300),
        (50, 80), (50, 320),
        (70, 60), (70, 340),
        (25, 180), (25, 220)
    ]
    scan_grid = manual_antenna_positions
    
    # 🔍 Валидация координат (защита от выхода за границы сетки)
    scan_grid = []
    h, w = temp_true.shape
    for pos in manual_antenna_positions:
        y, x = pos
        if 0 <= y < h and 0 <= x < w:
            # Опционально: привязка к ближайшему пикселю ткани
            if not breast_mask[y, x]:
                print(f"⚠️ Позиция ({y}, {x}) вне ткани. Ищем ближайшую...")
                # Простой поиск ближайшего True в радиусе 5 пикселей
                found = False
                for r in range(1, 6):
                    for dy in range(-r, r+1):
                        for dx in range(-r, r+1):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < h and 0 <= nx < w and breast_mask[ny, nx]:
                                scan_grid.append((ny, nx))
                                found = True
                                break
                        if found: break
                    if found: break
                if not found:
                    print(f"   ❌ Не удалось найти ткань рядом с ({y}, {x}). Пропущено.")
            else:
                scan_grid.append((y, x))
        else:
            print(f"⚠️ Позиция ({y}, {x}) вне границ сетки. Пропущено.")

    print(f"📡 Количество антенн на контуре: {len(scan_grid)}")
    '''
    print(f"Количество антенн: {len(scan_grid)}")
    print(f"Позиция сканера: Y={scan_y}")

    print("\nВыполнение прямого сканирования...")
    Tb_data, emissivity_avg = model.forward_scan(eps_map, cond_map, temp_true, breast_mask, scan_grid)

    print("\nДиагностика Tb (в Кельвинах):")
    print(f"   Мин Tb: {Tb_data.min():.2f} K ({Tb_data.min() - 273.15:.2f}°C)\n   Макс Tb: {Tb_data.max():.2f} K ({Tb_data.max() - 273.15:.2f}°C)\n   Среднее Tb: {Tb_data.mean():.2f} K ({Tb_data.mean() - 273.15:.2f}°C)\n   Emissivity: {emissivity_avg.mean():.3f} ± {emissivity_avg.std():.3f}")
    if Tb_data.min() < 260 or Tb_data.max() > 320:
        print("    ВНИМАНИЕ: Tb вне физиологического диапазона (260-320 K)!\n   Проверьте конвертацию °C → K в forward_scan()!")
    else: print("+ Tb в физиологическом диапазоне (260-320 K)")
    print("Emissivity в норме (0.90-0.99)" if emissivity_avg.mean() >= 0.90 else "   Emissivity слишком низкий!")
    print("Emissivity имеет вариацию" if emissivity_avg.std() >= 0.01 else "   Emissivity без вариации!")

    print("\nРеконструкция температуры...")
    Tb_noisy = Tb_data + np.random.normal(0, 0.05, size=Tb_data.shape)
    temp_recon = model.reconstruct_simple(Tb_noisy, emissivity_avg, scan_grid, temp_true.shape, breast_mask)

    print("\nГенерация графиков...")
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

    plot_antenna_coverage(breast_mask, scan_grid, model, temp_recon=temp_recon)

    print_full_statistics(temp_true, temp_recon, breast_mask, Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, model, tissue_type_map)

    total_elapsed = time.time() - start_total
    print(f"\nОбщее время выполнения: {total_elapsed:.2f} сек ({total_elapsed/60:.1f} мин)")
    print("\nВсе графики сохранены в файлы 01_*.png ... 11_*.png")
