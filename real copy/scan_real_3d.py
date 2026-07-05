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


def run_real_radiometry():
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
    
    scan_grid = hemispherical_array(
        shape, n_theta=5, n_phi=10, radius=55, air_buffer_z=25
    )
    print(f"   Антенн: {len(scan_grid)}")
    
    # 3. Сканирование
    print("\n[3/4] 📡 Прямое сканирование...")
    Tb_data, emissivity_avg = model.forward_scan_3d(
        temp_true, eps_map, breast_mask, scan_grid
    )
    
    if ADD_NOISE:
        noise = np.random.normal(0, NOISE_STD_K, size=Tb_data.shape)
        Tb_noisy = Tb_data + noise
    
    # 4. Реконструкция
    print("\n[4/4] 🔄 Реконструкция...")
    temp_recon = model.reconstruct_3d(
        Tb_noisy, emissivity_avg, scan_grid, shape, breast_mask
    )
    
    # Статистика
    valid_true = temp_true[breast_mask]
    valid_recon = temp_recon[breast_mask]
    mae = np.nanmean(np.abs(valid_true - valid_recon))
    rmse = np.sqrt(np.nanmean((valid_true - valid_recon)**2))
    
    print(f"\n📊 Результаты:")
    print(f"   RMSE: {rmse:.2f} °C")
    print(f"   MAE:  {mae:.2f} °C")
    
    # ============================================================
    # 🎨 ИНТЕРАКТИВНАЯ 3D ВИЗУАЛИЗАЦИЯ
    # ============================================================
    print("\n🎨 Подготовка интерактивной сцены...")
    
    # Транспонируем для PyVista
    temp_recon_grid = temp_recon.transpose(2, 1, 0)
    temp_true_grid = temp_true.transpose(2, 1, 0)
    breast_grid = breast_mask.transpose(2, 1, 0).astype(np.uint8)
    
    grid = pv.ImageData(dimensions=temp_recon_grid.shape)
    grid.point_data["temp_recon"] = temp_recon_grid.flatten(order="F")
    grid.point_data["temp_true"] = temp_true_grid.flatten(order="F")
    grid.point_data["breast"] = breast_grid.flatten(order="F")
    
    plotter = pv.Plotter(window_size=[1600, 1000])
    plotter.set_background("black")
    
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

    # ⭐ СТРЕЛКИ, показывающие направление "сверху вниз"
    # Вектор направления: вдоль оси Z PyVista (от антенн к груди)
    arrow_direction = np.array([0, 0, 1], dtype=np.float32)

    # ✅ ИСПРАВЛЕНИЕ: Размножаем вектор направления на каждую антенну
    # Создаем массив формы (N_antennas, 3), где каждая строка = [0, 0, 1]
    directions = np.tile(arrow_direction, (len(ant_points), 1))

    print(f"   📡 Точки антенн: shape={ant_points.shape}")
    print(f"   ➡️ Направления: shape={directions.shape}")

    # Добавляем стрелки на сцену
    plotter.add_arrows(
        ant_points,           # center: (N, 3)
        directions,           # direction: (N, 3) ← теперь правильной формы!
        mag=8.0,              # длина стрелки
        color="cyan",
        opacity=0.8,
        label="Направление антенн"
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
    
    clim = [34.0, 39.5]
    
    print("   Добавляю интерактивные слайсеры...")
    print("   💡 Используй мышь для перемещения плоскостей!")
    
    # Слайсер по Z (аксиальный)
    plotter.add_mesh_slice(
        grid,
        normal=[0, 0, 1],
        scalars="temp_recon",
        cmap="jet",
        clim=clim,
        opacity=0.95,
        name="slice_z",
        # Начальная позиция - через опухоль!
        origin=(grid.center[0], grid.center[1], tz if tumor_pos else grid.center[2])
    )
    
    # Слайсер по Y (корональный)
    plotter.add_mesh_slice(
        grid,
        normal=[0, 1, 0],
        scalars="temp_recon",
        cmap="jet",
        clim=clim,
        opacity=0.95,
        name="slice_y",
        origin=(grid.center[0], ty if tumor_pos else grid.center[1], grid.center[2])
    )
    
    # Слайсер по X (сагиттальный)
    plotter.add_mesh_slice(
        grid,
        normal=[1, 0, 0],
        scalars="temp_recon",
        cmap="jet",
        clim=clim,
        opacity=0.95,
        name="slice_x",
        origin=(tx if tumor_pos else grid.center[0], grid.center[1], grid.center[2])
    )
    
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
    print("   ⌨️ Клавиша 's' = сохранить скриншот")
    print("   ⌨️ Клавиша 'r' = сброс камеры")
    print("=" * 60)
    print("\n💡 ПОДСКАЗКА: Срезы уже установлены через опухоль!")
    print("   Попробуй подвигать их, чтобы увидеть полную картину.")
    
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