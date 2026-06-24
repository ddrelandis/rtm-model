import numpy as np
import pyvista as pv
from model_3d import BreastRadiometryModel3D

def run_3d_radiometry():
    print("="*50)
    print("🚀 ЗАПУСК 3D РАДИОТЕРМОМЕТРИИ")
    print("="*50)

    # 1. Инициализация и генерация фантома
    model = BreastRadiometryModel3D()
    shape = (80, 120, 120) # (z, y, x)
    eps, cond, temp_true, breast_mask, tumor_mask = model.create_anatomical_phantom(shape=shape, tumor_radius=12)

    # 2. Формирование 3D сетки антенн (планарная решетка над грудью)
    # Антенны располагаются на высоте z = -5 (чуть выше верхней границы груди)
    z_ant = -5 
    # Диапазон Y и X: от 20% до 80% размера сетки, чтобы покрыть грудь
    y_range = np.linspace(int(shape[1]*0.2), int(shape[1]*0.8), 8, dtype=int)
    x_range = np.linspace(int(shape[2]*0.2), int(shape[2]*0.8), 8, dtype=int)
    
    scan_grid_3d = []
    for y in y_range:
        for x in x_range:
            scan_grid_3d.append((z_ant, y, x))
            
    print(f"\n📡 Сформирована сетка антенн: {len(scan_grid_3d)} шт. (8x8)")

    # 3. Прямое сканирование (Forward Scan)
    print("\n[1/2] Выполнение прямого сканирования...")
    Tb_data = model.forward_scan_3d(temp_true, breast_mask, scan_grid_3d)
    print(f"   ✅ Tb (К): min={Tb_data.min():.2f}, max={Tb_data.max():.2f}, mean={Tb_data.mean():.2f}")

    # 4. Реконструкция (Back-projection)
    print("\n[2/2] Выполнение 3D реконструкции...")
    temp_recon = model.reconstruct_3d(Tb_data, scan_grid_3d, shape, breast_mask)

    # 5. Расчет 3D статистики (ошибок)
    valid_true = temp_true[breast_mask]
    valid_recon = temp_recon[breast_mask]
    
    abs_error = np.abs(valid_true - valid_recon)
    mae = abs_error.mean()
    rmse = np.sqrt(np.mean(abs_error**2))
    
    print("\n" + "="*50)
    print("📊 3D СТАТИСТИКА РЕКОНСТРУКЦИИ")
    print("="*50)
    print(f"   Истинная T (средняя):  {valid_true.mean():.2f} °C")
    print(f"   Реконструированная T:  {valid_recon.mean():.2f} °C")
    print(f"   MAE (Средняя ошибка):  {mae:.2f} °C")
    print(f"   RMSE (Среднеквадратич):{rmse:.2f} °C")
    print("="*50)

    # 6. 3D Визуализация результата в PyVista
    print("\n🎨 Подготовка 3D сцены для визуализации...")
    
    # Транспонируем в (x, y, z) для PyVista
    temp_recon_grid = temp_recon.transpose(2, 1, 0)
    tumor_grid = tumor_mask.transpose(2, 1, 0).astype(np.uint8)
    breast_grid = breast_mask.transpose(2, 1, 0).astype(np.uint8)

    grid = pv.ImageData(dimensions=temp_recon_grid.shape)
    grid.point_data["temp_recon"] = temp_recon_grid.flatten(order="F")
    grid.point_data["tumor"] = tumor_grid.flatten(order="F")
    grid.point_data["breast"] = breast_grid.flatten(order="F")

    plotter = pv.Plotter(window_size=[1200, 900])
    plotter.set_background("lightgray")

    # Контур груди
    breast_surf = grid.contour(isosurfaces=[0.5], scalars="breast")
    plotter.add_mesh(breast_surf, color="peachpuff", opacity=0.2, label="Молочная железа")

    # Контур истинной опухоли (для сравнения)
    tumor_surf = grid.contour(isosurfaces=[0.5], scalars="tumor")
    plotter.add_mesh(tumor_surf, color="black", opacity=0.3, label="Истинная опухоль (контур)")

    # Срезы реконструированной температуры
    bounds = grid.bounds
    z_mid = (bounds[4] + bounds[5]) / 2
    y_mid = (bounds[2] + bounds[3]) / 2
    
    clim = [34.0, 39.5]

    # Аксиальный срез (поперечный)
    slice_z = grid.slice(normal='z', origin=(0, 0, z_mid))
    plotter.add_mesh(slice_z, scalars="temp_recon", cmap="jet", clim=clim, opacity=0.95)

    # Корональный срез (фронтальный)
    slice_y = grid.slice(normal='y', origin=(0, y_mid, 0))
    plotter.add_mesh(slice_y, scalars="temp_recon", cmap="jet", clim=clim, opacity=0.95)

    # Добавляем маркеры антенн (красные точки над грудью)
    ant_points = np.array([[pos[2], pos[1], pos[0]] for pos in scan_grid_3d], dtype=np.float32)
    plotter.add_points(ant_points, color="red", point_size=10, label="Антенны (8x8)")

    plotter.add_axes()
    plotter.add_legend(loc='upper left')
    plotter.add_title(f"3D Реконструкция T | RMSE: {rmse:.2f}°C | MAE: {mae:.2f}°C", font_size=16, color="black")

    print("✅ Запуск интерактивного окна PyVista... (Закройте окно для завершения)")
    plotter.show()

if __name__ == "__main__":
    run_3d_radiometry()