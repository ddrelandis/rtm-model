"""
3D Радиотермометрия головного мозга
Главный скрипт запуска
"""

import numpy as np
import pyvista as pv
import os
import time
from scipy.ndimage import gaussian_filter

from model_brain import BrainRadiometryModel3D


# ============================================================
# ️ КОНФИГУРАЦИЯ
# ============================================================
QUALITY_PRESETS = {
    'low':    {'shape': (60, 80, 80),    'tumor_radius': 6,  'resolution_mm': 4},
    'medium': {'shape': (80, 120, 120),  'tumor_radius': 10, 'resolution_mm': 2},
    'high':   {'shape': (100, 160, 160), 'tumor_radius': 12, 'resolution_mm': 1.5},
    'ultra':  {'shape': (120, 200, 200), 'tumor_radius': 15, 'resolution_mm': 1},
}

QUALITY = 'medium'
FREQUENCY_GHZ = 2.0
ANTENNA_LAYOUT = 'helmet'
ADD_NOISE = True
NOISE_STD_K = 0.05
SAVE_SCREENSHOTS = False
OUTPUT_DIR = 'data_brain'
TUMOR_POS = None


# ============================================================
# 🔧 ГЕОМЕТРИИ АНТЕНН
# ============================================================

def helmet_antenna_array(shape, n_rings=3, n_ant_per_ring=16):
    """Шлемовидная решётка антенн вокруг головы."""
    d, h, w = shape
    center_y, center_x = h // 2, w // 2
    
    positions = []
    z_levels = np.linspace(int(d * 0.2), int(d * 0.8), n_rings)
    
    for z_lvl in z_levels:
        z_normalized = (z_lvl - d/2) / (d/2)
        ring_radius_y = int(h * 0.45 * np.sqrt(1 - z_normalized**2))
        ring_radius_x = int(w * 0.45 * np.sqrt(1 - z_normalized**2))
        
        for i in range(n_ant_per_ring):
            angle = 2 * np.pi * i / n_ant_per_ring
            y = int(center_y + ring_radius_y * np.cos(angle))
            x = int(center_x + ring_radius_x * np.sin(angle))
            
            if 0 <= y < h and 0 <= x < w:
                positions.append((int(z_lvl), y, x))
    
    return positions


def validate_antenna_positions(scan_positions, shape, mask):
    """Валидация позиций антенн."""
    d, h, w = shape
    valid_positions = []
    
    for pos in scan_positions:
        z, y, x = int(pos[0]), int(pos[1]), int(pos[2])
        if not (0 <= z < d and 0 <= y < h and 0 <= x < w):
            print(f"   ⚠️ Позиция ({z}, {y}, {x}) вне сетки. Пропущено.")
            continue
        valid_positions.append((z, y, x))
    
    print(f"   ✅ Принято позиций: {len(valid_positions)} из {len(scan_positions)}")
    return valid_positions


def print_brain_statistics(temp_true, temp_recon, head_mask, tumor_mask,
                           Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, structures):
    """Выводит полную статистику."""
    valid_true = temp_true[head_mask]
    valid_recon = temp_recon[head_mask]
    
    print("\n" + "=" * 60)
    print("📊 ПОЛНАЯ 3D СТАТИСТИКА (МОЗГ)")
    print("=" * 60)
    
    print(f"\n🔬 Диэлектрические свойства:")
    print(f"   ε (eps):    {np.mean(eps_map[head_mask]):6.2f} ± {np.std(eps_map[head_mask]):.2f}")
    print(f"   σ (cond):   {np.mean(cond_map[head_mask]):6.2f} ± {np.std(cond_map[head_mask]):.2f} См/м")
    
    print(f"\n🌡 Статистика температуры:")
    print(f"   Истинная T:          {valid_true.mean():6.2f} ± {valid_true.std():.2f} °C")
    print(f"   Реконструированная:  {np.nanmean(valid_recon):6.2f} ± {np.nanstd(valid_recon):.2f} °C")
    bias = np.nanmean(valid_recon) - valid_true.mean()
    print(f"   Смещение (bias):     {bias:+.2f} °C")
    
    abs_error = np.abs(valid_true - valid_recon)
    mae = np.nanmean(abs_error)
    rmse = np.sqrt(np.nanmean(abs_error**2))
    
    print(f"\n❌ Ошибки реконструкции:")
    print(f"   MAE:  {mae:.2f} °C")
    print(f"   RMSE: {rmse:.2f} °C")
    
    print(f"\n📡 Измерения:")
    print(f"   Антенн: {len(Tb_data)}")
    print(f"   Tb: {Tb_noisy.mean():.2f} K ({Tb_noisy.mean() - 273.15:.2f} °C)")
    print(f"   Emissivity: {emissivity_avg.mean():.3f} ± {emissivity_avg.std():.3f}")
    
    if np.any(tumor_mask):
        tumor_true = temp_true[tumor_mask]
        tumor_recon = temp_recon[tumor_mask]
        tumor_contrast = tumor_true.mean() - valid_true.mean()
        
        z_coords, y_coords, x_coords = np.where(tumor_mask)
        center_z = int(z_coords.mean())
        center_y = int(y_coords.mean())
        center_x = int(x_coords.mean())
        
        print(f"\n🎯 Опухоль:")
        print(f"   Центр: Z={center_z}, Y={center_y}, X={center_x}")
        print(f"   T (истина): {tumor_true.mean():.2f} °C")
        print(f"   T (рекон):  {np.nanmean(tumor_recon):.2f} °C")
        print(f"   Контраст:   {tumor_contrast:+.2f} °C")
        
        tumor_detected = np.nanmax(tumor_recon) > (valid_true.mean() + 0.5)
        print(f"   Обнаружена: {'✅ ДА' if tumor_detected else '❌ НЕТ'}")
    
    print(f"\n🧠 Структуры мозга:")
    for name, struct_mask in structures.items():
        if np.any(struct_mask & head_mask):
            t_mean = temp_true[struct_mask & head_mask].mean()
            t_recon = np.nanmean(temp_recon[struct_mask & head_mask])
            print(f"   {name:15s}: T_истина={t_mean:.2f}°C, T_рекон={t_recon:.2f}°C")
    
    print("=" * 60)
    
    return {'mae': mae, 'rmse': rmse, 'bias': bias, 'n_antennas': len(Tb_data)}


def create_brain_visualization(temp_recon, temp_true, head_mask, tumor_mask,
                               scan_positions, stats, quality_name, layout_name,
                               structures, save_screenshot=False, output_dir='data_brain'):
    """Создает 3D визуализацию мозга."""
    print("\n🎨 Подготовка 3D сцены...")
    
    temp_recon_grid = temp_recon.transpose(2, 1, 0)
    tumor_grid = tumor_mask.transpose(2, 1, 0).astype(np.uint8)
    head_grid = head_mask.transpose(2, 1, 0).astype(np.uint8)
    
    grid = pv.ImageData(dimensions=temp_recon_grid.shape)
    grid.point_data["temp_recon"] = temp_recon_grid.flatten(order="F")
    grid.point_data["tumor"] = tumor_grid.flatten(order="F")
    grid.point_data["head"] = head_grid.flatten(order="F")
    
    plotter = pv.Plotter(window_size=[1400, 900], off_screen=save_screenshot)
    plotter.set_background("lightgray")
    
    # 1. Полупрозрачный контур головы
    head_surf = grid.contour(isosurfaces=[0.5], scalars="head")
    if head_surf.n_points > 0:
        plotter.add_mesh(head_surf, color="lightblue", opacity=0.15, label="Голова")
    
    # 2. Контур истинной опухоли (только если есть)
    if np.any(tumor_mask):
        tumor_surf = grid.contour(isosurfaces=[0.5], scalars="tumor")
        if tumor_surf.n_points > 0:
            plotter.add_mesh(tumor_surf, color="red", opacity=0.6, label="Опухоль")
    else:
        print("   ⚠️ Опухоль пуста, пропускаем её визуализацию")
    
    # 3. Изосурфейс горячей зоны (только если есть)
    try:
        hot_zone = grid.threshold(value=38.0, scalars="temp_recon")
        if hot_zone.n_points > 0:
            hot_surf = hot_zone.contour(isosurfaces=[38.0], scalars="temp_recon")
            if hot_surf.n_points > 0:
                plotter.add_mesh(hot_surf, color="orange", opacity=0.7, 
                                label="Горячая зона (>38°C)")
    except Exception as e:
        print(f"   ⚠️ Не удалось построить горячую зону: {e}")
    
    bounds = grid.bounds
    z_mid = (bounds[4] + bounds[5]) / 2
    y_mid = (bounds[2] + bounds[3]) / 2
    
    slice_z = grid.slice(normal='z', origin=(0, 0, z_mid))
    plotter.add_mesh(slice_z, scalars="temp_recon", cmap="jet", clim=[34.0, 39.5], opacity=0.95)
    
    slice_y = grid.slice(normal='y', origin=(0, y_mid, 0))
    plotter.add_mesh(slice_y, scalars="temp_recon", cmap="jet", clim=[34.0, 39.5], opacity=0.95)
    
    z_levels = sorted(set(pos[0] for pos in scan_positions))
    colors = [pv.Color("blue"), pv.Color("green"), pv.Color("orange"), pv.Color("red")]
    
    for i, z_lvl in enumerate(z_levels):
        level_pts = np.array([[p[2], p[1], p[0]] for p in scan_positions if p[0] == z_lvl], dtype=np.float32)
        if len(level_pts) > 0:
            plotter.add_points(level_pts, color=colors[i % len(colors)], point_size=10,
                             label=f"Антенны Z={z_lvl}", render_points_as_spheres=True)
    
    plotter.add_axes()
    plotter.add_legend(loc='upper left')
    plotter.add_title(f"3D Мозг | {quality_name.upper()} | RMSE: {stats['rmse']:.2f}°C", font_size=14)
    
    if save_screenshot:
        os.makedirs(output_dir, exist_ok=True)
        plotter.show(auto_close=False)
        plotter.screenshot(f"{output_dir}/brain_{layout_name}_{quality_name}.png")
        plotter.close()
    else:
        plotter.show()


# ============================================================
# 🚀 ОСНОВНАЯ ФУНКЦИЯ
# ============================================================

def run_brain_radiometry():
    """Полный цикл 3D радиотермометрии мозга."""
    start_total = time.time()
    
    print("=" * 60)
    print("🚀 ЗАПУСК 3D РАДИОТЕРМОМЕТРИИ ГОЛОВНОГО МОЗГА")
    print("=" * 60)
    
    preset = QUALITY_PRESETS[QUALITY]
    print(f"\n⚙️ Режим: {QUALITY.upper()}")
    print(f"   Разрешение: {preset['shape']}")
    print(f"   Частота: {FREQUENCY_GHZ} ГГц")
    
    model = BrainRadiometryModel3D(freq_ghz=FREQUENCY_GHZ, resolution_mm=preset['resolution_mm'])
    
    print("\n[1/5] 🔬 Генерация фантома...")
    shape = preset['shape']
    eps_map, cond_map, temp_true, head_mask, tumor_mask, structures = \
        model.create_anatomical_brain_phantom(shape=shape, tumor_radius=preset['tumor_radius'], tumor_pos=TUMOR_POS)
    
    print(f"\n[2/5] 📡 Формирование антенн ({ANTENNA_LAYOUT})...")
    if ANTENNA_LAYOUT == 'helmet':
        scan_grid = helmet_antenna_array(shape, n_rings=3, n_ant_per_ring=16)
    else:
        raise ValueError(f"Неизвестная геометрия: {ANTENNA_LAYOUT}")
    
    print(f"   Сформировано: {len(scan_grid)}")
    scan_grid = validate_antenna_positions(scan_grid, shape, head_mask)
    
    if len(scan_grid) == 0:
        raise RuntimeError("Нет валидных позиций антенн!")
    
    print("\n[3/5]  Прямое сканирование...")
    Tb_data, emissivity_avg = model.forward_scan_3d(temp_true, eps_map, head_mask, scan_grid, 
                                                     skull_mask=structures.get('skull'))
    
    if ADD_NOISE:
        Tb_noisy = Tb_data + np.random.normal(0, NOISE_STD_K, size=Tb_data.shape)
        print(f"   🔊 Шум: σ = {NOISE_STD_K} K")
    else:
        Tb_noisy = Tb_data
    
    print(f"   ✅ Tb: {Tb_noisy.mean():.2f} K")
    
    print("\n[4/5] 🔄 Реконструкция...")
    temp_recon = model.reconstruct_3d(Tb_noisy, emissivity_avg, scan_grid, shape, head_mask,
                                       skull_mask=structures.get('skull'))
    
    print("\n[5/5] 📊 Статистика...")
    stats = print_brain_statistics(temp_true, temp_recon, head_mask, tumor_mask,
                                    Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, structures)
    
    create_brain_visualization(temp_recon, temp_true, head_mask, tumor_mask,
                                scan_grid, stats, QUALITY, ANTENNA_LAYOUT, structures,
                                save_screenshot=SAVE_SCREENSHOTS, output_dir=OUTPUT_DIR)
    
    total_time = time.time() - start_total
    print(f"\n⏱ Время: {total_time:.2f} сек ({total_time/60:.1f} мин)")
    
    return stats


if __name__ == "__main__":
    try:
        stats = run_brain_radiometry()
        print("\n✅ Завершено!")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()