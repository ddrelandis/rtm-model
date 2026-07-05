"""
Диагностика: проверка наличия и температуры каждого типа ткани.
Интерактивное включение/выключение слоев.
"""
import numpy as np
import pyvista as pv
import os

DATA_DIR = "data_real"

def main():
    # 1. Загрузка данных
    print("📂 Загрузка данных...")
    temp_map = np.load(os.path.join(DATA_DIR, "real_temp_map.npy"))
    tissue_type_map = np.load(os.path.join(DATA_DIR, "real_tissue_type_map.npy"))
    breast_mask = np.load(os.path.join(DATA_DIR, "real_breast_mask.npy"))
    
    shape = temp_map.shape
    print(f"   Размер: {shape}")
    
    # 2. Диагностика: что реально есть в массиве?
    print("\n" + "=" * 60)
    print("📊 ДИАГНОСТИКА ТКАНЕЙ В МАССИВЕ")
    print("=" * 60)
    
    tissue_names = {
        0: 'Фон (воздух)',
        1: 'Жир (fat)',
        2: 'Кровь (clot)',
        3: 'Кожа (skin)',
        4: 'Железа (gland)'
    }
    
    for tid, name in tissue_names.items():
        mask = (tissue_type_map == tid)
        count = np.sum(mask)
        if count > 0:
            t_vals = temp_map[mask]
            print(f"   ✅ {name:20s}: {count:>8,} вокселей | "
                  f"T: {t_vals.mean():5.2f}°C "
                  f"(min={t_vals.min():.2f}, max={t_vals.max():.2f})")
        else:
            print(f"   ❌ {name:20s}: НЕТ ВОКСЕЛЕЙ!")
    
    print("=" * 60)
    
    # Проверка: есть ли вообще разница температур между тканями?
    fat_mask = (tissue_type_map == 1)
    clot_mask = (tissue_type_map == 2)
    skin_mask = (tissue_type_map == 3)
    
    if np.any(fat_mask) and np.any(clot_mask):
        diff = temp_map[clot_mask].mean() - temp_map[fat_mask].mean()
        print(f"\n   🔍 Разница clot-fat: {diff:+.2f}°C")
        if abs(diff) < 0.5:
            print("   ⚠️ ВНИМАНИЕ: Разница меньше 0.5°C — ткани неразличимы!")
    if np.any(fat_mask) and np.any(skin_mask):
        diff = temp_map[skin_mask].mean() - temp_map[fat_mask].mean()
        print(f"   🔍 Разница skin-fat: {diff:+.2f}°C")
        if abs(diff) < 0.5:
            print("   ⚠️ ВНИМАНИЕ: Разница меньше 0.5°C — ткани неразличимы!")
    
    # 3. Интерактивная визуализация с переключением слоев
    print("\n🎨 Подготовка интерактивной визуализации...")
    
    # Транспонируем для PyVista
    temp_grid = temp_map.transpose(2, 1, 0).astype(np.float32)
    tissue_grid = tissue_type_map.transpose(2, 1, 0).astype(np.float32)
    breast_grid = breast_mask.transpose(2, 1, 0).astype(np.uint8)
    
    grid = pv.ImageData(dimensions=temp_grid.shape)
    grid.point_data["temp"] = temp_grid.flatten(order="F")
    grid.point_data["tissue"] = tissue_grid.flatten(order="F")
    grid.point_data["breast"] = breast_grid.flatten(order="F")
    
    plotter = pv.Plotter(window_size=[1400, 900])
    plotter.set_background("black")
    
    # Создаем отдельные меши для каждого типа ткани
    meshes = {}
    colors = {
        1: ('yellow', 'Жир (fat)'),
        2: ('red', 'Кровь (clot)'),
        3: ('pink', 'Кожа (skin)'),
        4: ('tan', 'Железа (gland)'),
    }
    
    for tid, (color, label) in colors.items():
        # Извлекаем только воксели этого типа
        tissue_binary = (tissue_grid == tid).astype(np.uint8)
        
        # Создаём временную сетку с маской ткани
        temp_grid_tissue = grid.copy()
        temp_grid_tissue.point_data["mask"] = tissue_binary.flatten(order="F")
        
        # Threshold: оставляем только воксели этого типа
        try:
            tissue_mesh = temp_grid_tissue.threshold(value=0.5, scalars="mask")
            if tissue_mesh.n_points > 0:
                meshes[tid] = tissue_mesh
                plotter.add_mesh(
                    tissue_mesh,
                    color=color,
                    opacity=0.6,
                    label=label,
                    name=f"tissue_{tid}"
                )
                print(f"   ✅ Добавлен слой: {label} ({tissue_mesh.n_points} точек)")
            else:
                print(f"   ❌ Слой пуст: {label}")
        except Exception as e:
            print(f"   ❌ Ошибка добавления {label}: {e}")
    
    # Полупрозрачный контур груди
    breast_surf = grid.contour(isosurfaces=[0.5], scalars="breast")
    plotter.add_mesh(breast_surf, color="white", opacity=0.1, 
                    label="Контур груди", name="breast_contour")
    
    # Срез с температурой (фон)
    bounds = grid.bounds
    z_mid = (bounds[4] + bounds[5]) / 2
    slice_z = grid.slice(normal='z', origin=(0, 0, z_mid))
    plotter.add_mesh(slice_z, scalars="temp", cmap="jet", 
                    clim=[30.0, 42.0], opacity=0.7, name="temp_slice")
    
    # Инструкции
    instructions = (
        "TOGGLE LAYERS:\n"
        "  1 = Fat (yellow)\n"
        "  2 = Blood/clot (red)\n"
        "  3 = Skin (pink)\n"
        "  4 = Gland (tan)\n"
        "  0 = Show all\n"
        "  r = Reset camera"
    )
    plotter.add_text(instructions, position="lower_left", font_size=10, 
                    color="white", name="instructions")
    
    plotter.add_axes()
    plotter.add_legend(loc='upper left')
    plotter.add_title("DIAGNOSTICS: Toggle tissue layers with keys 1-4", 
                     font_size=14, color="white")
    
    # Привязываем клавиши к переключению видимости
    def toggle_layer(tid):
        actor_name = f"tissue_{tid}"
        if actor_name in plotter.renderer.actors:
            actor = plotter.renderer.actors[actor_name]
            # Переключаем видимость
            current_vis = actor.GetVisibility()
            actor.SetVisibility(not current_vis)
            status = "ON" if not current_vis else "OFF"
            print(f"   🔘 {colors[tid][1]}: {status}")
            plotter.render()
    
    plotter.add_key_event('1', lambda: toggle_layer(1))
    plotter.add_key_event('2', lambda: toggle_layer(2))
    plotter.add_key_event('3', lambda: toggle_layer(3))
    plotter.add_key_event('4', lambda: toggle_layer(4))
    
    def show_all():
        for tid in colors:
            actor_name = f"tissue_{tid}"
            if actor_name in plotter.renderer.actors:
                plotter.renderer.actors[actor_name].SetVisibility(True)
        print("   🔘 Все слои включены")
        plotter.render()
    
    plotter.add_key_event('0', show_all)
    
    print("\n" + "=" * 60)
    print("🎮 УПРАВЛЕНИЕ")
    print("=" * 60)
    print("   Клавиша 1 = Жир (вкл/выкл)")
    print("   Клавиша 2 = Кровь (вкл/выкл)")
    print("   Клавиша 3 = Кожа (вкл/выкл)")
    print("   Клавиша 4 = Железа (вкл/выкл)")
    print("   Клавиша 0 = Показать все")
    print("=" * 60)
    
    plotter.show()


if __name__ == "__main__":
    main()