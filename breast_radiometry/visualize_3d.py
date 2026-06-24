import pyvista as pv
import numpy as np
from model_3d import BreastRadiometryModel3D

def visualize_3d_phantom():
    print("🔄 Генерация 3D фантома для визуализации...")
    model = BreastRadiometryModel3D()
    # Генерируем фантом (можно увеличить разрешение, если ПК тянет)
    eps, cond, temp, breast_mask, tumor_mask = model.create_anatomical_phantom(shape=(80, 120, 120))

    print("📦 Подготовка данных для PyVista...")
    # PyVista требует порядок осей (x, y, z), поэтому транспонируем из (z, y, x)
    temp_grid = temp.transpose(2, 1, 0)
    breast_grid = breast_mask.transpose(2, 1, 0).astype(np.uint8)
    tumor_grid = tumor_mask.transpose(2, 1, 0).astype(np.uint8)

    # Создаем 3D-сетку (Uniform Grid)
    grid = pv.ImageData(dimensions=temp_grid.shape)
    # PyVista требует, чтобы данные были "сплюснуты" (flattened) в порядке Fortran ('F')
    grid.point_data["temperature"] = temp_grid.flatten(order="F")
    grid.point_data["breast"] = breast_grid.flatten(order="F")
    grid.point_data["tumor"] = tumor_grid.flatten(order="F")

    print("🎨 Запуск 3D-окна визуализации...")
    plotter = pv.Plotter(window_size=[1000, 800])
    plotter.set_background("white")

    # 1. Полупрозрачная поверхность молочной железы (Изосурфейс)
    # Мы "вытягиваем" 3D-меш из бинарной маски груди
    breast_surf = grid.contour(isosurfaces=[0.5], scalars="breast")
    plotter.add_mesh(breast_surf, color="peachpuff", opacity=0.25, label="Молочная железа")

    # 2. Поверхность опухоли (Ярко-красная)
    tumor_surf = grid.contour(isosurfaces=[0.5], scalars="tumor")
    plotter.add_mesh(tumor_surf, color="red", opacity=0.8, label="Опухоль")

    # 3. Ортогональные срезы (Слайсеры) с температурой
    # Находим центр координат для срезов
    bounds = grid.bounds
    x_mid = (bounds[0] + bounds[1]) / 2
    y_mid = (bounds[2] + bounds[3]) / 2
    z_mid = (bounds[4] + bounds[5]) / 2

    # Настройки цветовой шкалы (от 34 до 39.5 градусов)
    clim = [34.0, 39.5]

    # Срез по оси Z (Аксиальный / поперечный)
    slice_z = grid.slice(normal='z', origin=(0, 0, z_mid))
    plotter.add_mesh(slice_z, scalars="temperature", cmap="jet", clim=clim, opacity=0.9)

    # Срез по оси Y (Сагиттальный / боковой)
    slice_y = grid.slice(normal='y', origin=(0, y_mid, 0))
    plotter.add_mesh(slice_y, scalars="temperature", cmap="jet", clim=clim, opacity=0.9)

    # Срез по оси X (Корональный / фронтальный)
    slice_x = grid.slice(normal='x', origin=(x_mid, 0, 0))
    plotter.add_mesh(slice_x, scalars="temperature", cmap="jet", clim=clim, opacity=0.9)

    # Интерактивный слайсер (можно двигать мышкой прямо в окне!)
    # plotter.add_mesh_slice(grid, scalars="temperature", cmap="jet", clim=clim)

    # Добавляем оси координат, легенду и заголовок
    plotter.add_axes()
    plotter.add_legend(loc='upper left')
    plotter.add_title("3D Фантом МЖ: Температурное поле и Опухоль", font_size=16, color="black")

    # Запуск интерактивного окна
    plotter.show()

if __name__ == "__main__":
    visualize_3d_phantom()