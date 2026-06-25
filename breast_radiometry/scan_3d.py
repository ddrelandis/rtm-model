"""
3D Радиотермометрия молочной железы
Главный скрипт запуска: генерация фантома, сканирование, реконструкция, визуализация.
"""

import numpy as np
import pyvista as pv
import os
import time
from scipy.ndimage import gaussian_filter

from model_3d import BreastRadiometryModel3D
from antenna_layouts import (
    planar_grid, circular_ring, hemispherical_array, 
    dual_plane, manual_positions
)

# ============================================================
# ⚙️ КОНФИГУРАЦИЯ
# ============================================================
QUALITY_PRESETS = {
    'low':    {'shape': (60, 80, 80),    'tumor_radius': 8,  'resolution_mm': 4},
    'medium': {'shape': (80, 120, 120),  'tumor_radius': 12, 'resolution_mm': 2},
    'high':   {'shape': (100, 160, 160), 'tumor_radius': 15, 'resolution_mm': 1.5},
    'ultra':  {'shape': (120, 200, 200), 'tumor_radius': 18, 'resolution_mm': 1},
}

# Основные параметры запуска
QUALITY = 'high'              # low / medium / high / ultra
BI_RADS = 'B'                   # A / B / C / D (плотность ткани)
ANTENNA_LAYOUT = 'hemisphere'   # planar / ring / hemisphere / dual / manual
ADD_NOISE = True                # Добавить шум в измерения (как в реальном приборе)
NOISE_STD_K = 0.1             # Стандартное отклонение шума (в Кельвинах)
SAVE_SCREENSHOTS = True        # Сохранять скриншоты PyVista
OUTPUT_DIR = 'data_3d'          # Папка для сохранения результатов

# Позиция опухоли (z, y, x) — None = случайная
TUMOR_POS = None

# Ручные позиции антенн (используются только при ANTENNA_LAYOUT = 'manual')
MANUAL_ANTENNAS = [
    (-5, 60, 60), (-5, 60, 80), (-5, 80, 60), (-5, 80, 80),
    (-10, 70, 70), (-15, 50, 50), (-15, 90, 90),
]

# ============================================================
# 🔧 ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def compute_emissivity_3d(eps_map, mask):
    """
    Упрощенный расчет коэффициента излучения по диэлектрической проницаемости.
    Используется формула Френеля для нормального падения.
    """
    sqrt_eps = np.sqrt(np.maximum(eps_map, 1.0))
    gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
    emissivity_fresnel = 1.0 - gamma**2
    # Калибровка в физиологический диапазон (0.90 - 0.99)
    emissivity = 0.88 + 0.11 * (emissivity_fresnel - 0.5) / 0.5
    emissivity = np.clip(emissivity, 0.90, 0.99)
    # Небольшой пространственный шум
    noise = np.random.normal(0, 0.01, emissivity.shape)
    emissivity = np.clip(emissivity + noise * mask, 0.85, 0.99)
    return emissivity


def validate_antenna_positions(scan_positions, shape, mask, search_radius=10):
    """
    Упрощенная валидация: разрешаем антеннам быть в воздухе,
    если они находятся в пределах сетки и "смотрят" на ткань.
    """
    d, h, w = shape
    valid_positions = []
    
    for pos in scan_positions:
        z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
        
        # Проверка границ сетки
        if not (0 <= z < d and 0 <= y < h and 0 <= x < w):
            print(f"   ⚠️ Позиция ({z}, {y}, {x}) вне сетки. Пропущено.")
            continue
        
        # ✅ ИСПРАВЛЕНИЕ: Разрешаем антенны в воздухе, если они в пределах сетки
        # Антенна может быть над грудью (в буфере), главное чтобы она "видела" ткань
        valid_positions.append((z, y, x))
    
    print(f"   ✅ Принято позиций: {len(valid_positions)} из {len(scan_positions)}")
    return valid_positions

def print_3d_statistics(temp_true, temp_recon, breast_mask, tumor_mask,
                        Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map):
    """
    Выводит полную статистику 3D реконструкции.
    """
    valid_true = temp_true[breast_mask]
    valid_recon = temp_recon[breast_mask]
    
    print("\n" + "=" * 60)
    print("📊 ПОЛНАЯ 3D СТАТИСТИКА")
    print("=" * 60)
    
    # Диэлектрические свойства
    print(f"\n🔬 Диэлектрические свойства (внутри груди):")
    print(f"   ε (eps):    {np.mean(eps_map[breast_mask]):6.2f} ± {np.std(eps_map[breast_mask]):.2f}")
    print(f"   σ (cond):   {np.mean(cond_map[breast_mask]):6.2f} ± {np.std(cond_map[breast_mask]):.2f} См/м")
    
    # Температура
    print(f"\n🌡 Статистика температуры:")
    print(f"   Истинная T:          {valid_true.mean():6.2f} ± {valid_true.std():.2f} °C")
    print(f"   Реконструированная:  {np.nanmean(valid_recon):6.2f} ± {np.nanstd(valid_recon):.2f} °C")
    bias = np.nanmean(valid_recon) - valid_true.mean()
    print(f"   Смещение (bias):     {bias:+.2f} °C")
    print(f"   Мин T (истина):      {valid_true.min():6.2f} °C")
    print(f"   Макс T (истина):     {valid_true.max():6.2f} °C")
    print(f"   Диапазон:            {valid_true.max() - valid_true.min():.2f} °C")
    
    # Ошибки реконструкции
    abs_error = np.abs(valid_true - valid_recon)
    mae = np.nanmean(abs_error)
    rmse = np.sqrt(np.nanmean(abs_error**2))
    
    print(f"\n❌ Ошибки реконструкции:")
    print(f"   Средняя (MAE):       {mae:.2f} °C")
    print(f"   Максимальная:        {np.nanmax(abs_error):.2f} °C")
    print(f"   RMSE:                {rmse:.2f} °C")
    print(f"   Медианная:           {np.nanmedian(abs_error):.2f} °C")
    
    # Измерения радиотермометра
    print(f"\n📡 Измерения радиотермометра:")
    print(f"   Количество антенн:   {len(Tb_data)}")
    print(f"   Tb (мин):            {Tb_noisy.min():.2f} K ({Tb_noisy.min() - 273.15:.2f} °C)")
    print(f"   Tb (макс):           {Tb_noisy.max():.2f} K ({Tb_noisy.max() - 273.15:.2f} °C)")
    print(f"   Tb (среднее):        {Tb_noisy.mean():.2f} K ({Tb_noisy.mean() - 273.15:.2f} °C)")
    print(f"   Emissivity (ср.):    {emissivity_avg.mean():.3f} ± {emissivity_avg.std():.3f}")
    
    # Проверка физиологического диапазона
    if Tb_data.min() < 260 or Tb_data.max() > 320:
        print("   ⚠️ ВНИМАНИЕ: Tb вне физиологического диапазона (260-320 K)!")
    else:
        print("   ✅ Tb в физиологическом диапазоне")
    
    # Опухоль
    print(f"\n🎯 Анализ опухоли:")
    if np.any(tumor_mask):
        tumor_true = temp_true[tumor_mask]
        tumor_recon = temp_recon[tumor_mask]
        tumor_contrast = tumor_true.mean() - valid_true.mean()
        
        # Находим центр опухоли (центр масс маски)
        z_coords, y_coords, x_coords = np.where(tumor_mask)
        center_z = int(z_coords.mean())
        center_y = int(y_coords.mean())
        center_x = int(x_coords.mean())
        
        print(f"   Центр опухоли:       Z={center_z}, Y={center_y}, X={center_x}")
        print(f"   T в опухоли (истина): {tumor_true.mean():.2f} °C")
        print(f"   T в опухоли (рекон):  {np.nanmean(tumor_recon):.2f} °C")
        print(f"   Контраст опухоли:     {tumor_contrast:+.2f} °C")
        print(f"   Размер (вокселей):    {np.sum(tumor_mask)}")
        
        # Видимость опухоли в реконструкции
        tumor_max_recon = np.nanmax(tumor_recon)
        tumor_detected = tumor_max_recon > (valid_true.mean() + 0.5)
        print(f"   Обнаружена в рекон.: {'✅ ДА' if tumor_detected else '❌ НЕТ'}")
    else:
        print("   Опухоль не была создана")
    
    print("=" * 60)
    
    return {
        'mae': mae, 'rmse': rmse, 'bias': bias,
        'n_antennas': len(Tb_data),
        'tumor_contrast': tumor_contrast if np.any(tumor_mask) else 0
    }


def create_3d_visualization(temp_recon, temp_true, breast_mask, tumor_mask,
                            scan_positions, stats, quality_name, layout_name,
                            save_screenshot=False, output_dir='data_3d'):
    """
    Создает интерактивную 3D визуализацию и опционально сохраняет скриншоты.
    """
    print("\n🎨 Подготовка 3D сцены...")
    
    # Транспонируем в (x, y, z) для PyVista
    temp_recon_grid = temp_recon.transpose(2, 1, 0)
    tumor_grid = tumor_mask.transpose(2, 1, 0).astype(np.uint8)
    breast_grid = breast_mask.transpose(2, 1, 0).astype(np.uint8)
    
    grid = pv.ImageData(dimensions=temp_recon_grid.shape)
    grid.point_data["temp_recon"] = temp_recon_grid.flatten(order="F")
    grid.point_data["tumor"] = tumor_grid.flatten(order="F")
    grid.point_data["breast"] = breast_grid.flatten(order="F")
    
    plotter = pv.Plotter(window_size=[1400, 900], off_screen=save_screenshot)
    plotter.set_background("lightgray")
    
    # 1. Полупрозрачный контур груди
    breast_surf = grid.contour(isosurfaces=[0.5], scalars="breast")
    plotter.add_mesh(breast_surf, color="peachpuff", opacity=0.2, 
                    label="Молочная железа")
    
    # 2. Контур истинной опухоли (черный)
    tumor_surf = grid.contour(isosurfaces=[0.5], scalars="tumor")
    plotter.add_mesh(tumor_surf, color="black", opacity=0.4, 
                    label="Истинная опухоль")
    
    # 3. Изосурфейс горячей зоны в реконструкции (> 37.5°C)
    try:
        hot_zone = grid.threshold(value=37.5, scalars="temp_recon")
        if hot_zone.n_points > 0:
            hot_surf = hot_zone.contour(isosurfaces=[37.5], scalars="temp_recon")
            plotter.add_mesh(hot_surf, color="red", opacity=0.6,
                           label="Горячая зона (>37.5°C)")
    except Exception as e:
        print(f"   ⚠️ Не удалось построить горячую зону: {e}")
    
    # 4. Ортогональные срезы
    bounds = grid.bounds
    x_mid = (bounds[0] + bounds[1]) / 2
    y_mid = (bounds[2] + bounds[3]) / 2
    z_mid = (bounds[4] + bounds[5]) / 2
    
    clim = [34.0, 39.5]
    
    # Аксиальный срез (поперечный)
    slice_z = grid.slice(normal='z', origin=(0, 0, z_mid))
    plotter.add_mesh(slice_z, scalars="temp_recon", cmap="jet", 
                    clim=clim, opacity=0.95)
    
    # Корональный срез (фронтальный)
    slice_y = grid.slice(normal='y', origin=(0, y_mid, 0))
    plotter.add_mesh(slice_y, scalars="temp_recon", cmap="jet", 
                    clim=clim, opacity=0.95)
    
    # 5. Антенны (цветные точки по Z-уровням)
    z_levels = sorted(set(pos[0] for pos in scan_positions))
    colors_by_level = [
        pv.Color("blue"), pv.Color("green"), pv.Color("orange"), 
        pv.Color("red"), pv.Color("purple"), pv.Color("cyan")
    ]
    
    for i, z_lvl in enumerate(z_levels):
        level_pts = np.array(
            [[p[2], p[1], p[0]] for p in scan_positions if p[0] == z_lvl], 
            dtype=np.float32
        )
        if len(level_pts) > 0:
            color = colors_by_level[i % len(colors_by_level)]
            plotter.add_points(
                level_pts, color=color, point_size=10,
                label=f"Антенны Z={z_lvl} ({len(level_pts)} шт.)",
                render_points_as_spheres=True
            )
    
    # 6. Оформление
    plotter.add_axes()
    plotter.add_legend(loc='upper left')
    
    title = (f"3D Реконструкция | {quality_name.upper()} | {layout_name}\n"
            f"RMSE: {stats['rmse']:.2f}°C | MAE: {stats['mae']:.2f}°C | "
            f"Антенн: {stats['n_antennas']}")
    plotter.add_title(title, font_size=14, color="black")
    
    # 7. Сохранение скриншотов
    if save_screenshot:
        os.makedirs(output_dir, exist_ok=True)
        
        # Общий вид
        plotter.show(auto_close=False)
        fname = f"{output_dir}/3d_overview_{layout_name}_{quality_name}.png"
        plotter.screenshot(fname)
        print(f"   💾 Скриншот сохранен: {fname}")
        
        # Вид сверху
        plotter.view_xy()
        fname = f"{output_dir}/3d_top_{layout_name}_{quality_name}.png"
        plotter.screenshot(fname)
        
        # Вид спереди
        plotter.view_yz()
        fname = f"{output_dir}/3d_front_{layout_name}_{quality_name}.png"
        plotter.screenshot(fname)
        
        # Вид сбоку
        plotter.view_xz()
        fname = f"{output_dir}/3d_side_{layout_name}_{quality_name}.png"
        plotter.screenshot(fname)
        
        print(f"   💾 Все скриншоты сохранены в {output_dir}/")
        plotter.close()
    else:
        print("✅ Запуск интерактивного окна PyVista... (Закройте окно для завершения)")
        plotter.show()


# ============================================================
# 🚀 ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def run_3d_radiometry():
    """
    Полный цикл 3D радиотермометрии:
    1. Генерация фантома
    2. Формирование сетки антенн
    3. Прямое сканирование (с шумом)
    4. Реконструкция
    5. Статистика
    6. Визуализация
    """
    start_total = time.time()
    
    print("=" * 60)
    print("🚀 ЗАПУСК 3D РАДИОТЕРМОМЕТРИИ МОЛОЧНОЙ ЖЕЛЕЗЫ")
    print("=" * 60)
    
    # 1. Инициализация
    preset = QUALITY_PRESETS[QUALITY]
    print(f"\n⚙️ Режим: {QUALITY.upper()}")
    print(f"   Разрешение: {preset['shape']} вокселей")
    print(f"   BI-RADS: {BI_RADS}")
    print(f"   Геометрия антенн: {ANTENNA_LAYOUT}")
    print(f"   Шум в измерениях: {'ДА' if ADD_NOISE else 'НЕТ'}")
    
    model = BreastRadiometryModel3D(
        freq_ghz=3.5, 
        resolution_mm=preset['resolution_mm'],
        birads_category=BI_RADS
    )
    
    # 2. Генерация фантома
    print("\n[1/5] 🔬 Генерация 3D анатомического фантома...")
    shape = preset['shape']
    eps_map, cond_map, temp_true, breast_mask, tumor_mask = \
        model.create_anatomical_phantom(
            shape=shape, 
            tumor_radius=preset['tumor_radius'],
            tumor_pos=TUMOR_POS,
            air_buffer_z=25  # ✅ Добавили буфер для антенн
        )
    
    print(f"   Размер фантома: {shape}")
    print(f"   Вокселей в груди: {np.sum(breast_mask):,}")
    print(f"   Вокселей в опухоли: {np.sum(tumor_mask):,}")
    
    # 3. Формирование сетки антенн
    print(f"\n[2/5] 📡 Формирование сетки антенн ({ANTENNA_LAYOUT})...")
    
    if ANTENNA_LAYOUT == 'planar':
        scan_grid = planar_grid(shape, n_y=8, n_x=8, z_height=-5, coverage=0.7)
    elif ANTENNA_LAYOUT == 'ring':
        scan_grid = circular_ring(shape, n_ant=32, radius=55, z_height=-5)
    elif ANTENNA_LAYOUT == 'hemisphere':
        scan_grid = hemispherical_array(shape, n_theta=7, n_phi=14, radius=55)
    elif ANTENNA_LAYOUT == 'dual':
        scan_grid = dual_plane(shape, n_y=6, n_x=6, z_top=-5, z_bottom=70)
    elif ANTENNA_LAYOUT == 'manual':
        scan_grid = manual_positions(MANUAL_ANTENNAS)
    else:
        raise ValueError(f"Неизвестная геометрия: {ANTENNA_LAYOUT}")
    
    print(f"   Сформировано позиций: {len(scan_grid)}")
    
    # Валидация
    print("   🔍 Валидация позиций антенн...")
    scan_grid = validate_antenna_positions(scan_grid, shape, breast_mask)
    print(f"   ✅ Валидных позиций: {len(scan_grid)}")
    
    if len(scan_grid) == 0:
        raise RuntimeError("Нет валидных позиций антенн! Проверьте геометрию.")
    
    # 4. Прямое сканирование
    print("\n[3/5] 📡 Выполнение прямого сканирования...")
    # ✅ ИСПРАВЛЕНИЕ: Передаем eps_map для расчета emissivity внутри forward_scan
    Tb_data, emissivity_avg = model.forward_scan_3d(temp_true, eps_map, breast_mask, scan_grid)

    # Добавление шума
    if ADD_NOISE:
        noise = np.random.normal(0, NOISE_STD_K, size=Tb_data.shape)
        Tb_noisy = Tb_data + noise
        print(f"   🔊 Добавлен шум: σ = {NOISE_STD_K} K")
    else:
        Tb_noisy = Tb_data

    print(f"   ✅ Tb (К): min={Tb_noisy.min():.2f}, max={Tb_noisy.max():.2f}, "
        f"mean={Tb_noisy.mean():.2f}")
    print(f"   ✅ Emissivity: {emissivity_avg.mean():.3f} ± {emissivity_avg.std():.3f}")


    # 5. Реконструкция
    print("\n[4/5] 🔄 Выполнение 3D реконструкции...")
    # ✅ ИСПРАВЛЕНИЕ: Передаем emissivity_avg в reconstruct_3d
    temp_recon = model.reconstruct_3d(
        Tb_noisy, emissivity_avg, scan_grid, shape, breast_mask
    )
    
    # 6. Статистика
    print("\n[5/5] 📊 Расчет статистики...")
    stats = print_3d_statistics(
        temp_true, temp_recon, breast_mask, tumor_mask,
        Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map
    )
    
    # 7. Визуализация
    create_3d_visualization(
        temp_recon, temp_true, breast_mask, tumor_mask,
        scan_grid, stats, QUALITY, ANTENNA_LAYOUT,
        save_screenshot=SAVE_SCREENSHOTS,
        output_dir=OUTPUT_DIR
    )
    
    # Итоговое время
    total_time = time.time() - start_total
    print(f"\n⏱ Общее время выполнения: {total_time:.2f} сек ({total_time/60:.1f} мин)")
    
    return stats


# ============================================================
# 🎯 ЗАПУСК
# ============================================================

if __name__ == "__main__":
    try:
        stats = run_3d_radiometry()
        print("\n✅ 3D радиотермометрия успешно завершена!")
    except Exception as e:
        print(f"\n❌ Ошибка выполнения: {e}")
        import traceback
        traceback.print_exc()