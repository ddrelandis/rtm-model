"""
3D Визуализация диаграммы направленности антенны
Показывает функцию чувствительности антенны в объёме молочной железы
"""
import numpy as np
import pyvista as pv
import os
from model_3d import BreastRadiometryModel3D
from antenna_layouts import hemispherical_array

def visualize_radiation_pattern_3d(
    ant_position=None,
    shape=(80, 120, 120),
    output_path='data_3d/radiation_pattern_3d.png',
    show_interactive=True
):
    """
    Создает 3D визуализацию диаграммы направленности антенны.
    
    Параметры:
    - ant_position: позиция антенны (z, y, x) или None для автоматического выбора
    - shape: размер сетки
    - output_path: путь для сохранения скриншота
    - show_interactive: показать интерактивное окно
    """

    
    # генерация фантома
    print("\n[1/4]Генерация 3D фантома...")
    model = BreastRadiometryModel3D()
    eps, cond, temp, breast_mask, tumor_mask = model.create_anatomical_phantom(
        shape=shape, tumor_radius=12, air_buffer_z=25
    )
    
    # формирование сетки антенн
    print("\n[2/4] Формирование сетки антенн...")
    scan_grid = hemispherical_array(shape, n_theta=5, n_phi=10, radius=55, air_buffer_z=25)
    
    # выбираем антенну
    if ant_position is None:
        ant_idx = len(scan_grid) // 2
        ant_position = scan_grid[ant_idx]
        print(f"Выбрана антенна #{ant_idx}: Z={ant_position[0]}, Y={ant_position[1]}, X={ant_position[2]}")
    else:
        print(f"Используется заданная позиция: Z={ant_position[0]}, Y={ant_position[1]}, X={ant_position[2]}")
    
    # вычисление ядра чувствительности
    print("\n[3/4]Вычисление диаграммы направленности...")
    kernel = model.compute_sensitivity_kernel_3d(breast_mask, ant_position)
    
    print(f"Размер ядра: {kernel.shape}")
    print(f"Максимум чувствительности: {kernel.max():.6f}")
    print(f"Сумма весов: {kernel.sum():.4f}")
    
    # визуализация
    print("\n[4/4] Построение 3D сцены...")
    
    kernel_grid = kernel.transpose(2, 1, 0)
    breast_grid = breast_mask.transpose(2, 1, 0).astype(np.uint8)
    
    grid = pv.ImageData(dimensions=kernel_grid.shape)
    grid.point_data["sensitivity"] = kernel_grid.flatten(order="F")
    grid.point_data["breast"] = breast_grid.flatten(order="F")
    
    plotter = pv.Plotter(window_size=[1400, 1000], off_screen=not show_interactive)
    plotter.set_background("white")
    
    # полупрозрачный контур груди
    breast_surf = grid.contour(isosurfaces=[0.5], scalars="breast")
    plotter.add_mesh(breast_surf, color="peachpuff", opacity=0.15, 
                    label="Молочная железа")
    
    # изосурфейсы чувствительности (разные уровни)
    levels = [0.001, 0.005, 0.01, 0.02]
    colors = ['blue', 'cyan', 'yellow', 'red']
    opacities = [0.1, 0.2, 0.3, 0.4]
    
    for level, color, opacity in zip(levels, colors, opacities):
        try:
            iso = grid.contour(isosurfaces=[level], scalars="sensitivity")
            if iso.n_points > 0:
                plotter.add_mesh(iso, color=color, opacity=opacity, 
                               label=f"Уровень {level:.3f}")
        except Exception as e:
            print(f"Не удалось построить изосурфейс {level}: {e}")
    
    # ортогональные срезы через центр антенны
    bounds = grid.bounds
    ant_x, ant_y, ant_z = ant_position[2], ant_position[1], ant_position[0]
    
    # срез XY (поперечный) - на уровне антенны
    slice_xy = grid.slice(normal='z', origin=(0, 0, ant_z))
    plotter.add_mesh(slice_xy, scalars="sensitivity", cmap="hot", 
                    opacity=0.6, clim=[0, kernel.max() * 0.5])
    
    # срез XZ (сагиттальный) - через центр антенны
    slice_xz = grid.slice(normal='y', origin=(0, ant_y, 0))
    plotter.add_mesh(slice_xz, scalars="sensitivity", cmap="hot", 
                    opacity=0.6, clim=[0, kernel.max() * 0.5])
    
    # срез YZ (корональный) - через центр антенны
    slice_yz = grid.slice(normal='x', origin=(ant_x, 0, 0))
    plotter.add_mesh(slice_yz, scalars="sensitivity", cmap="hot", 
                    opacity=0.6, clim=[0, kernel.max() * 0.5])
    
    # позиция антенны (большая красная сфера)
    ant_point = np.array([[ant_x, ant_y, ant_z]], dtype=np.float32)
    plotter.add_points(ant_point, color="red", point_size=25, 
                      render_points_as_spheres=True, label="Антенна")
    
    # стрелка направления 
    arrow_start = [ant_x, ant_y, ant_z]
    arrow_direction = [0, 0, 1]
    arrow = pv.Arrow(start=arrow_start, direction=arrow_direction, scale=30.0)
    plotter.add_mesh(arrow, color="red", opacity=0.8, label="Направление")
    
    # все антенны 
    all_ant_points = np.array([[p[2], p[1], p[0]] for p in scan_grid], dtype=np.float32)
    plotter.add_points(all_ant_points, color="blue", point_size=5, 
                      render_points_as_spheres=True, label="Все антенны", opacity=0.5)

    for pos in scan_grid:
        ant_start = [pos[2], pos[1], pos[0]]
        ant_direction = [0, 0, 1]
        arrow = pv.Arrow(start=ant_start, direction=ant_direction, scale=3.0)
        plotter.add_mesh(arrow, color="blue", opacity=0.4)

    plotter.add_axes()
    
    try:
        plotter.add_legend(loc='upper left')
    except ValueError:
        print("Легенда пропущена (нет элементов с метками)")
    
    title = (f"3D Диаграмма направленности антенны\n"
            f"Позиция: Z={ant_position[0]}, Y={ant_position[1]}, X={ant_position[2]}")
    plotter.add_title(title, font_size=14, color="black")
    

    if show_interactive:
        print("\nЗапуск интерактивного окна PyVista...")
        plotter.show()
    else:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plotter.show(auto_close=False)
        plotter.screenshot(output_path)
        plotter.close()
        print(f"\nСкриншот сохранен: {output_path}")
    
    return kernel


if __name__ == "__main__":
    # 3D визуализация диаграммы направленности
    kernel = visualize_radiation_pattern_3d(
        ant_position=None,
        shape=(80, 120, 120),
        output_path='data_3d/radiation_pattern_3d.png',
        show_interactive=True
    )
