"""
3D Радиотермометрия на реальных данных МРТ
ИНТЕРАКТИВНАЯ ВЕРСИЯ с движущимися срезами
"""
import numpy as np
import pyvista as pv
import os
import time
from model_3d import BreastRadiometryModel3D
from antenna_layouts import hemispherical_array

# === НАСТРОЙКИ ===
DATA_DIR = "data_real"
ANTENNA_LAYOUT = 'hemisphere'
ADD_NOISE = True
NOISE_STD_K = 0.05
OUTPUT_DIR = 'results_real'


def run_real_radiometry(**kwargs):
    print("=" * 60)
    print("🚀 ИНТЕРАКТИВНАЯ РАДИОТЕРМОМЕТРИЯ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 60)
    
    # 1. Загрузка
    print("\n[1/4] 📂 Загрузка реальных данных...")
    eps_map = np.load(os.path.join(DATA_DIR, "real_eps_map.npy"))
    temp_true = np.load(os.path.join(DATA_DIR, "real_temp_map.npy"))
    breast_mask = np.load(os.path.join(DATA_DIR, "real_breast_mask.npy"))
    metadata = np.load(os.path.join(DATA_DIR, "real_metadata.npy"), allow_pickle=True).item()
    
    shape = temp_true.shape
    tumor_pos = metadata['tumor_pos']
    
    print(f"   Размер: {shape}")
    print(f"   Опухоль: Z={tumor_pos[0]}, Y={tumor_pos[1]}, X={tumor_pos[2]}")
    print(f"   ⚠️ Опухоль находится на Z={tumor_pos[0]} из {shape[0]} (близко к границе!)")
    
    # 2. Модель и антенны
    print("\n[2/4] 📡 Формирование сетки антенн...")
    model = BreastRadiometryModel3D(freq_ghz=3.5, resolution_mm=2.0)
    
    use_custom = os.environ.get('USE_CUSTOM_ANTENNAS') == '1'
    custom_config_path = os.path.join(DATA_DIR, "antenna_config.npy")
    
    # ⭐ Переменные для расчёта и визуализации
    antenna_directions_array = None  # для model.forward_scan (координаты массива z,y,x)
    antenna_directions_pv = None     # для визуализации (координаты PyVista x,y,z)
    
    if use_custom and os.path.exists(custom_config_path):
        print(f"   🎯 Используется пользовательская конфигурация")
        config = np.load(custom_config_path, allow_pickle=True).item()
        
        scan_grid = []
        dirs_array_list = []
        dirs_pv_list = []
        
        for ant in config['antennas']:
            x_pv, y_pv, z_pv = ant['pos_pyvista']
            scan_grid.append((int(round(z_pv)), int(round(y_pv)), int(round(x_pv))))
            
            # Направление для расчёта (координаты массива z,y,x)
            dir_vec = np.array(ant['direction_vector'], dtype=np.float32)
            dirs_array_list.append(np.array([dir_vec[2], dir_vec[1], dir_vec[0]], dtype=np.float32))
            
            # Направление для визуализации (координаты PyVista x,y,z) - как есть
            dirs_pv_list.append(dir_vec)
        
        antenna_directions_array = dirs_array_list
        antenna_directions_pv = np.array(dirs_pv_list, dtype=np.float32)
        
        print(f"   Загружено антенн из файла: {len(scan_grid)}")
        
        # Статистика по направлениям
        unique_dirs = set(tuple(d) for d in antenna_directions_array)
        print(f"   🧭 Уникальных направлений: {len(unique_dirs)}")
    else:
        scan_grid = hemispherical_array(
            shape, n_theta=5, n_phi=10, radius=55, air_buffer_z=25
        )
        print(f"   Используется стандартная полусфера: {len(scan_grid)} антенн")
        
        # Дефолт для полусферы: все антенны смотрят вниз (+Z в PyVista)
        antenna_directions_pv = np.tile(
            np.array([0, 0, 1], dtype=np.float32), 
            (len(scan_grid), 1)
        )
    
    print(f"   Антенн: {len(scan_grid)}")

    # 3. Сканирование
    print("\n[3/4] 📡 Прямое сканирование...")
    Tb_data, emissivity_avg = model.forward_scan_3d(
        temp_true, eps_map, breast_mask, scan_grid,
        antenna_directions=antenna_directions_array  # ← Передаём направления
    )
    
    if ADD_NOISE:
        noise = np.random.normal(0, NOISE_STD_K, size=Tb_data.shape)
        Tb_noisy = Tb_data + noise
    
    # 4. Реконструкция
    print("\n[4/4] 🔄 Реконструкция...")
    temp_recon = model.reconstruct_3d(
        Tb_noisy, emissivity_avg, scan_grid, shape, breast_mask,
        antenna_directions=antenna_directions_array  # ← Передаём направления
    )
    
    # Статистика
    valid_true = temp_true[breast_mask]
    valid_recon = temp_recon[breast_mask]
    mae = np.nanmean(np.abs(valid_true - valid_recon))
    rmse = np.sqrt(np.nanmean((valid_true - valid_recon)**2))
    
    print(f"\n📊 Результаты:")
    print(f"   RMSE: {rmse:.2f} °C")
    print(f"   MAE:  {mae:.2f} °C")
    
    # ⭐ ДОБАВЛЕНА ПОДРОБНАЯ ДИАГНОСТИКА ПО ТКАНЯМ
    tissue_type_map = np.load(os.path.join(DATA_DIR, "real_tissue_type_map.npy"))
    tissue_names = {1: 'Жир (fat)', 2: 'Кровь (clot)', 
                    3: 'Кожа (skin)', 4: 'Железа (gland)'}
    
    print(f"\n🔬 Температура по тканям (СРАВНЕНИЕ ИСТИНА vs РЕКОНСТРУКЦИЯ):")
    print(f"   {'Ткань':<15} {'T_true (°C)':<25} {'T_recon (°C)':<25} {'Δ (°C)':<10}")
    print("   " + "-" * 75)
    
    for tid, name in tissue_names.items():
        tmask = (tissue_type_map == tid)
        if np.sum(tmask) > 0:
            t_true_mean = temp_true[tmask].mean()
            t_true_min = temp_true[tmask].min()
            t_true_max = temp_true[tmask].max()
            
            t_recon_mean = np.nanmean(temp_recon[tmask])
            t_recon_min = np.nanmin(temp_recon[tmask])
            t_recon_max = np.nanmax(temp_recon[tmask])
            
            delta = t_recon_mean - t_true_mean
            print(f"   {name:<15} {t_true_mean:5.2f} [{t_true_min:.2f}-{t_true_max:.2f}]  "
                  f"{t_recon_mean:5.2f} [{t_recon_min:.2f}-{t_recon_max:.2f}]  "
                  f"{delta:+.2f}")
    
    # ⭐ АВТОМАТИЧЕСКИЙ clim на основе реального диапазона
    clim_true = [temp_true[breast_mask].min(), temp_true[breast_mask].max()]
    clim_recon = [np.nanmin(temp_recon[breast_mask]), np.nanmax(temp_recon[breast_mask])]
    
    print(f"\n🎨 Цветовые диапазоны:")
    print(f"   clim_true:  [{clim_true[0]:.2f}, {clim_true[1]:.2f}] °C")
    print(f"   clim_recon: [{clim_recon[0]:.2f}, {clim_recon[1]:.2f}] °C")
    print(f"   ⚠️ Использую clim_true для визуализации — так видны ВСЕ ткани!")
    
    # ============================================================
    # 🎨 ИНТЕРАКТИВНАЯ 3D ВИЗУАЛИЗАЦИЯ
    # ============================================================
    print("\n🎨 Подготовка интерактивной сцены...")
    
    # Транспонируем для PyVista
    temp_recon_grid = temp_recon.transpose(2, 1, 0)
    temp_true_grid = temp_true.transpose(2, 1, 0)
    breast_grid = breast_mask.transpose(2, 1, 0).astype(np.uint8)
    tissue_grid = tissue_type_map.transpose(2, 1, 0).astype(np.uint8)
    
    grid = pv.ImageData(dimensions=temp_recon_grid.shape)
    grid.point_data["temp_recon"] = temp_recon_grid.flatten(order="F")
    grid.point_data["temp_true"] = temp_true_grid.flatten(order="F")
    grid.point_data["breast"] = breast_grid.flatten(order="F")
    grid.point_data["tissue"] = tissue_grid.flatten(order="F")  # ⭐ Добавили типы тканей
    
    plotter = pv.Plotter(window_size=[1600, 1000])
    plotter.set_background("black")
    
    # ⭐ ТЕКУЩИЙ РЕЖИМ ОТОБРАЖЕНИЯ (можно переключать клавишей 't')
    current_mode = {'scalars': 'temp_true', 'clim': clim_true}
    
    # 1. Полупрозрачный контур груди
    breast_surf = grid.contour(isosurfaces=[0.5], scalars="breast")
    plotter.add_mesh(breast_surf, color="peachpuff", opacity=0.15, 
                    label="Молочная железа")
    
    # 2. Антенны со стрелками направления "сверху вниз"
    ant_points = np.array([[p[2], p[1], p[0]] for p in scan_grid], dtype=np.float32)

    # Сферы на позициях антенн
    plotter.add_points(
        ant_points, 
        color="cyan", 
        point_size=8, 
        render_points_as_spheres=True,
        label="Антенны"
    )

    # ⭐ СТРЕЛКИ С ИНДИВИДУАЛЬНЫМИ НАПРАВЛЕНИЯМИ
    if antenna_directions_pv is not None and len(antenna_directions_pv) == len(ant_points):
        plotter.add_arrows(
            ant_points,
            direction=antenna_directions_pv,  # ← Каждая стрелка со своим направлением!
            mag=10.0,                          # Чуть длиннее, чтобы было хорошо видно
            color="lime",
            opacity=0.9,
            label="Направления антенн"
        )
        print(f"   🧭 Визуализация: {len(antenna_directions_pv)} индивидуальных направлений")
    else:
        # Fallback (на случай если что-то пошло не так)
        fallback_dir = np.tile(np.array([0, 0, 1], dtype=np.float32), (len(ant_points), 1))
        plotter.add_arrows(
            ant_points,
            direction=fallback_dir,
            mag=8.0,
            color="cyan",
            opacity=0.8,
            label="Направление (дефолт)"
        )
    
    # 3. Маркер опухоли (ИСХОДНАЯ позиция)
    if tumor_pos:
        tz, ty, tx = tumor_pos
        tumor_point = np.array([[tx, ty, tz]], dtype=np.float32)
        plotter.add_points(tumor_point, color="yellow", point_size=20,
                          render_points_as_spheres=True, 
                          label="Истинная опухоль")
    
    # 4. ⭐ ИНТЕРАКТИВНЫЕ СРЕЗЫ (slice widgets)
    # Это главная фича - ты сможешь двигать плоскости мышкой!
    
        # ⭐ АВТОМАТИЧЕСКИЙ clim на основе ИСТИННОЙ температуры
    clim = clim_true
    print(f"   🎨 clim для срезов: [{clim[0]:.2f}, {clim[1]:.2f}] °C")
    
    # Функция для создания слайсера (чтобы переиспользовать)
    def add_slice(normal, origin, name):
        plotter.add_mesh_slice(
            grid,
            normal=normal,
            scalars=current_mode['scalars'],  # temp_true по умолчанию
            cmap="jet",
            clim=clim,
            opacity=0.95,
            name=name,
            show_scalar_bar=True,  # ⭐ Цветовая шкала видна!
        )
    
    # Слайсер по Z (аксиальный)
    add_slice([0, 0, 1], 
              (grid.center[0], grid.center[1], tz if tumor_pos else grid.center[2]),
              "slice_z")
    
    # Слайсер по Y (корональный)
    add_slice([0, 1, 0],
              (grid.center[0], ty if tumor_pos else grid.center[1], grid.center[2]),
              "slice_y")
    
    # Слайсер по X (сагиттальный)
    add_slice([1, 0, 0],
              (tx if tumor_pos else grid.center[0], grid.center[1], grid.center[2]),
              "slice_x")
    
    # ⭐ ДОПОЛНИТЕЛЬНАЯ ФУНКЦИЯ: Переключение true/recon по клавише 't'
    def toggle_mode():
        if current_mode['scalars'] == 'temp_true':
            current_mode['scalars'] = 'temp_recon'
            current_mode['clim'] = clim_recon
            mode_name = "RECONSTRUCTED"
        else:
            current_mode['scalars'] = 'temp_true'
            current_mode['clim'] = clim_true
            mode_name = "TRUE"
        
        # Удаляем старые слайсеры
        for name in ["slice_z", "slice_y", "slice_x"]:
            try:
                plotter.remove_actor(name)
            except:
                pass
        
        # Добавляем новые с актуальными данными
        add_slice([0, 0, 1], 
                  (grid.center[0], grid.center[1], tz if tumor_pos else grid.center[2]),
                  "slice_z")
        add_slice([0, 1, 0],
                  (grid.center[0], ty if tumor_pos else grid.center[1], grid.center[2]),
                  "slice_y")
        add_slice([1, 0, 0],
                  (tx if tumor_pos else grid.center[0], grid.center[1], grid.center[2]),
                  "slice_x")
        
        print(f"   🔄 Режим: {mode_name} | clim: [{current_mode['clim'][0]:.2f}, {current_mode['clim'][1]:.2f}]")
    
    plotter.add_key_event('t', toggle_mode)
    
    # 5. ⭐ АВТОМАТИЧЕСКИЙ СРЕЗ ЧЕРЕЗ ОПУХОЛЬ (дополнительно)
    if tumor_pos:
        print(f"\n🎯 Добавляю автоматический срез через опухоль (Z={tz})...")
        tumor_slice = grid.slice(normal='z', origin=(0, 0, tz))
        plotter.add_mesh(
            tumor_slice,
            scalars="temp_true",  # Истинная температура
            cmap="hot",
            clim=clim,
            opacity=0.8,
            name="tumor_slice",
            show_scalar_bar=False
        )
    
    # 6. Горячая зона (с адаптивным порогом)
    tumor_region_temp = temp_true[tz-3:tz+3, ty-3:ty+3, tx-3:tx+3] if tumor_pos else None
    if tumor_region_temp is not None:
        hot_threshold = np.percentile(temp_true[breast_mask], 95)
        print(f"   Порог горячей зоны: {hot_threshold:.2f}°C")
        try:
            hot_zone = grid.threshold(value=hot_threshold, scalars="temp_true")
            if hot_zone.n_points > 0:
                plotter.add_mesh(
                    hot_zone, color="red", opacity=0.4,
                    label=f"Горячая зона (>{hot_threshold:.1f}°C)"
                )
        except Exception as e:
            print(f"   ⚠️ Не удалось построить горячую зону: {e}")
    
    # 7. Оформление
    plotter.add_axes()
    plotter.add_legend(loc='upper left')
    plotter.add_title(
        f"ИНТЕРАКТИВНАЯ 3D | RMSE: {rmse:.2f}°C | MAE: {mae:.2f}°C\n"
        f"💡 Двигай плоскости мышкой | Опухоль: Z={tz}, Y={ty}, X={tx}",
        font_size=12, color="white"
    )
    
    # 8. Инструкции
    print("\n" + "=" * 60)
    print("🎮 УПРАВЛЕНИЕ")
    print("=" * 60)
    print("   🖱 Левая кнопка + движение = вращение модели")
    print("   🔘 Колесико = приближение/отдаление")
    print("   📏 Наведи на плоскость + ЛКМ = ДВИГАТЬ СРЕЗ")
    print("   ⌨️ Клавиша 't' = переключить TRUE ↔ RECONSTRUCTED")
    print("   ⌨️ Клавиша 's' = сохранить скриншот")
    print("   ⌨️ Клавиша 'r' = сброс камеры")
    print("=" * 60)
    print("\n💡 ПОДСКАЗКА:")
    print("   Сейчас показываются ИСТИННЫЕ температуры (все ткани видны).")
    print("   Нажми 't', чтобы переключиться на РЕКОНСТРУИРОВАННЫЕ.")
    
    # Сохранение скриншотов
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    plotter.show(auto_close=False)
    
    # Автоматические скриншоты через опухоль
    if tumor_pos:
        plotter.screenshot(os.path.join(OUTPUT_DIR, "real_tumor_view.png"))
        print(f"\n💾 Скриншот через опухоль: {OUTPUT_DIR}/real_tumor_view.png")
    
    plotter.close()
    print("\n✅ Завершено!")


if __name__ == "__main__":
    try:
        run_real_radiometry()
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()