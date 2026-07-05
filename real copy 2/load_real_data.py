import numpy as np
import SimpleITK as sitk
import pandas as pd
import pyvista as pv
import os

class RealBreastDataLoader:
    """Загружает и подготавливает реальные данные МРТ из 3D Slicer"""
    
    def __init__(self, nrrd_path, labels_csv_path):
        self.nrrd_path = nrrd_path
        self.labels_csv_path = labels_csv_path
        self.volume = None
        self.segmentation = None
        self.label_map = {}
        self.spacing = None
        
    def load_labels(self):
        """
        Универсальный парсер CSV из 3D Slicer.
        Автоматически определяет колонки с ID и названием ткани.
        """
        print("📋 Загрузка меток тканей...")
        
        try:
            df = pd.read_csv(self.labels_csv_path)
        except Exception as e:
            print(f"   ❌ Ошибка чтения CSV: {e}")
            # Fallback: ручной словарь
            self.label_map = {1: 'fat', 2: 'clot', 3: 'skin'}
            print(f"   ⚠️ Используется fallback словарь: {self.label_map}")
            return self.label_map
        
        print(f"   Найдено строк: {len(df)}")
        print(f"   Колонки в CSV: {list(df.columns)}")
        print(f"   Первые строки:")
        print(df.head())
        print()
        
        # Автоматический поиск колонки с ID
        id_col = None
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(k in col_lower for k in ['label', 'id', 'value', 'index']):
                id_col = col
                break
        
        # Автоматический поиск колонки с именем
        name_col = None
        for col in df.columns:
            col_lower = str(col).lower().strip()
            if any(k in col_lower for k in ['name', 'tissue', 'segment', 'label_name']):
                name_col = col
                break
        
        # Если не нашли — берем первые две колонки
        if id_col is None:
            id_col = df.columns[0]
            print(f"   ⚠️ ID колонка не найдена автоматически, использую: '{id_col}'")
        if name_col is None:
            name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
            print(f"   ⚠️ Name колонка не найдена автоматически, использую: '{name_col}'")
        
        print(f"   ✅ Используемые колонки: ID='{id_col}', Name='{name_col}'")
        
        # Заполнение словаря
        for _, row in df.iterrows():
            try:
                label_id = int(row[id_col])
                name = str(row[name_col]).lower().strip()
                
                # Пропускаем фон (обычно label=0 или пустое имя)
                if label_id == 0 or not name or name == 'background':
                    continue
                    
                self.label_map[label_id] = name
                print(f"   {label_id}: {name}")
            except (ValueError, TypeError) as e:
                print(f"   ⚠️ Пропущена строка: {row.values} ({e})")
        
        # Если ничего не нашли — создаем дефолтный словарь
        if not self.label_map:
            print("   ⚠️ Словарь пуст! Создаю дефолтный на основе имен файлов...")
            # Fallback: пытаемся угадать по типичным именам
            default_map = {}
            for _, row in df.iterrows():
                try:
                    label_id = int(row[df.columns[0]])
                    name = str(row[df.columns[1]]).lower().strip() if len(df.columns) > 1 else f"tissue_{label_id}"
                    if label_id > 0:
                        default_map[label_id] = name
                except:
                    pass
            
            if default_map:
                self.label_map = default_map
            else:
                self.label_map = {1: 'fat', 2: 'clot', 3: 'skin'}
        
        print(f"\n✅ Итоговый словарь: {self.label_map}")
        return self.label_map

    def load_volume(self):
        """Загружает оригинальный МРТ объем"""
        print(f"\n📂 Загрузка МРТ: {self.nrrd_path}")
        
        image = sitk.ReadImage(self.nrrd_path)
        self.volume = sitk.GetArrayFromImage(image).astype(np.float32)
        self.spacing = image.GetSpacing()  # (x, y, z) в мм
        
        print(f"   Размер: {self.volume.shape} (Z, Y, X)")
        print(f"   Spacing: {self.spacing} мм")
        print(f"   Диапазон интенсивности: [{self.volume.min():.1f}, {self.volume.max():.1f}]")
        
        return self.volume
    
    def load_segmentation(self, seg_nrrd_path):
        """Загружает сегментацию"""
        print(f"\n🎯 Загрузка сегментации: {seg_nrrd_path}")
        
        seg_image = sitk.ReadImage(seg_nrrd_path)
        self.segmentation = sitk.GetArrayFromImage(seg_image).astype(np.int32)
        
        print(f"   Размер: {self.segmentation.shape}")
        print(f"   Уникальные метки: {np.unique(self.segmentation)}")
        
        # Анализ тканей
        for label_id, name in self.label_map.items():
            count = np.sum(self.segmentation == label_id)
            if count > 0:
                volume_mm3 = count * np.prod(self.spacing)
                volume_ml = volume_mm3 / 1000
                print(f"   {name}: {count:,} вокселей ({volume_ml:.1f} мл)")
        
        return self.segmentation
    
    def create_tissue_maps(self):
        """
        Создает карты диэлектрических свойств и температуры
        на основе сегментации.
        """
        print("\n🔬 Генерация физических свойств тканей...")
        
        shape = self.segmentation.shape
        eps_map = np.ones(shape, dtype=np.float32)  # Воздух = 1.0
        cond_map = np.zeros(shape, dtype=np.float32)
        temp_map = np.ones(shape, dtype=np.float32) * 20.0  # Воздух = 20°C
        tissue_type_map = np.zeros(shape, dtype=np.int32)
        
        # Физические свойства тканей (из твоего model.py)
        tissue_props = {
            'fat': {'eps': 5.0, 'std_eps': 0.5, 'cond': 0.10, 'std_cond': 0.03, 'temp': 35.0, 'type_id': 1},
            'clot': {'eps': 55.0, 'std_eps': 5.0, 'cond': 3.5, 'std_cond': 0.5, 'temp': 37.0, 'type_id': 2},  # Кровь
            'skin': {'eps': 35.0, 'std_eps': 4.0, 'cond': 1.0, 'std_cond': 0.2, 'temp': 33.8, 'type_id': 3},
        }
        
        # Создаем маску груди (всё, что не фон)
        breast_mask = self.segmentation > 0
        
        # Заполнение тканей
        for label_id, name in self.label_map.items():
            if label_id == 0:
                continue  # Пропускаем фон
                
            name_lower = name.lower()
            
            # Находим соответствующие свойства
            props = None
            for tissue_name, tissue_data in tissue_props.items():
                if tissue_name in name_lower:
                    props = tissue_data
                    break
            
            if props is None:
                print(f"   ⚠️ Неизвестная ткань: {name}, пропускаем")
                continue
            
            # Маска текущей ткани
            mask = self.segmentation == label_id
            count = np.sum(mask)
            
            if count == 0:
                continue
            
            # Заполнение физическими свойствами с шумом
            eps_map[mask] = np.random.normal(props['eps'], props['std_eps'], count)
            cond_map[mask] = np.random.normal(props['cond'], props['std_cond'], count)
            temp_map[mask] = props['temp']
            tissue_type_map[mask] = props['type_id']
            
            print(f"   ✅ {name}: ε={props['eps']:.1f}, σ={props['cond']:.2f}, T={props['temp']:.1f}°C")
        
        # Добавляем железистую ткань (если её нет в сегментации)
        # Это эвристика: всё внутри груди, что не жир/кожа/сосуды = железа
        gland_mask = breast_mask & ~np.isin(tissue_type_map, [1, 2, 3])
        if np.sum(gland_mask) > 0:
            gland_count = np.sum(gland_mask)
            eps_map[gland_mask] = np.random.normal(45.0, 5.0, gland_count)
            cond_map[gland_mask] = np.random.normal(2.4, 0.4, gland_count)
            temp_map[gland_mask] = 35.5
            tissue_type_map[gland_mask] = 4  # Железа
            print(f"   ✅ Добавлена железистая ткань (автоматически): {gland_count:,} вокселей")
        
        # Температурный градиент от поверхности вглубь
        from scipy.ndimage import distance_transform_edt, gaussian_filter
        
        dist_from_surface = distance_transform_edt(~breast_mask).astype(np.float32)
        dist_from_surface[~breast_mask] = 0
        max_dist = dist_from_surface[breast_mask].max()
        if max_dist > 0:
            normalized_depth = dist_from_surface / max_dist
            temp_map += 1.5 * (normalized_depth ** 0.6) * breast_mask
        
        # Сглаживание
        temp_map = gaussian_filter(temp_map, sigma=1.5)
        eps_map = gaussian_filter(eps_map, sigma=1.5)
        cond_map = gaussian_filter(cond_map, sigma=1.5)
        
        # Ограничение диапазона
        temp_map = np.clip(temp_map, 34.0, 39.5)
        temp_map[~breast_mask] = 20.0
        eps_map[~breast_mask] = 1.0
        cond_map[~breast_mask] = 0.0
        
        print(f"\n✅ Карты созданы:")
        print(f"   breast_mask: {np.sum(breast_mask):,} вокселей")
        print(f"   T диапазон: [{temp_map[breast_mask].min():.2f}, {temp_map[breast_mask].max():.2f}]°C")
        
        return eps_map, cond_map, temp_map, breast_mask, tissue_type_map
    
    def add_virtual_tumor(self, temp_map, breast_mask, tissue_type_map, 
                         tumor_radius_voxels=15, tumor_temp_increase=3.0):
        """
        Добавляет виртуальную опухоль в случайную позицию железистой ткани.
        """
        print(f"\n🎯 Добавление виртуальной опухоли (R={tumor_radius_voxels} вокселей)...")
        
        # Ищем позиции для опухоли (в железистой ткани или жире)
        valid_mask = (tissue_type_map == 4) | (tissue_type_map == 1)  # Железа или жир
        valid_z, valid_y, valid_x = np.where(valid_mask)
        
        if len(valid_z) == 0:
            print("   ❌ Не найдено подходящих позиций для опухоли!")
            return temp_map, None
        
        # Выбираем случайную позицию
        idx = np.random.randint(0, len(valid_z))
        tumor_z, tumor_y, tumor_x = valid_z[idx], valid_y[idx], valid_x[idx]
        
        print(f"   Позиция опухоли: Z={tumor_z}, Y={tumor_y}, X={tumor_x}")
        
        # Создаем маску опухоли (3D сфера)
        d, h, w = temp_map.shape
        z, y, x = np.ogrid[:d, :h, :w]
        dist_to_tumor = np.sqrt((z - tumor_z)**2 + (y - tumor_y)**2 + (x - tumor_x)**2)
        tumor_mask = dist_to_tumor <= tumor_radius_voxels
        
        # Повышаем температуру в опухоли
        tumor_sigma = tumor_radius_voxels * 1.5
        temp_map += tumor_temp_increase * np.exp(-dist_to_tumor**2 / (2 * tumor_sigma**2)) * breast_mask
        
        # Воспалительная зона вокруг опухоли
        inflammation_radius = tumor_radius_voxels * 2.5
        temp_map += 0.8 * np.exp(-dist_to_tumor**2 / (2 * inflammation_radius**2)) * breast_mask
        
        # Ограничение
        temp_map = np.clip(temp_map, 34.0, 39.5)
        
        print(f"   ✅ Опухоль добавлена: +{tumor_temp_increase:.1f}°C")
        print(f"   Макс T после: {temp_map[breast_mask].max():.2f}°C")
        
        return temp_map, (tumor_z, tumor_y, tumor_x)
    
    def visualize(self, temp_map, breast_mask, tumor_pos=None):
        """3D визуализация через PyVista"""
        print("\n🎨 Визуализация...")
        
        # Транспонируем для PyVista (z,y,x -> x,y,z)
        temp_grid = temp_map.transpose(2, 1, 0)
        breast_grid = breast_mask.transpose(2, 1, 0).astype(np.uint8)
        
        grid = pv.ImageData(dimensions=temp_grid.shape)
        grid.point_data["temperature"] = temp_grid.flatten(order="F")
        grid.point_data["breast"] = breast_grid.flatten(order="F")
        
        plotter = pv.Plotter(window_size=[1400, 900])
        plotter.set_background("black")
        
        # Контур груди
        breast_surf = grid.contour(isosurfaces=[0.5], scalars="breast")
        plotter.add_mesh(breast_surf, color="peachpuff", opacity=0.3, label="Молочная железа")
        
        # Горячая зона (>37.5°C)
        try:
            hot_zone = grid.threshold(value=37.5, scalars="temperature")
            if hot_zone.n_points > 0:
                hot_surf = hot_zone.contour(isosurfaces=[37.5], scalars="temperature")
                plotter.add_mesh(hot_surf, color="red", opacity=0.7, label="Горячая зона")
        except:
            pass
        
        # Срезы
        bounds = grid.bounds
        z_mid = (bounds[4] + bounds[5]) / 2
        y_mid = (bounds[2] + bounds[3]) / 2
        
        slice_z = grid.slice(normal='z', origin=(0, 0, z_mid))
        plotter.add_mesh(slice_z, scalars="temperature", cmap="jet", 
                        clim=[34.0, 39.5], opacity=0.9)
        
        slice_y = grid.slice(normal='y', origin=(0, y_mid, 0))
        plotter.add_mesh(slice_y, scalars="temperature", cmap="jet", 
                        clim=[34.0, 39.5], opacity=0.9)
        
        # Маркер опухоли
        # Маркер опухоли
    if tumor_pos:
        tz, ty, tx = tumor_pos
        tumor_point = np.array([[tx, ty, tz]], dtype=np.float32)  # ← Явное преобразование
        plotter.add_points(tumor_point, color="yellow", point_size=20, 
                        render_points_as_spheres=True, label="Опухоль")
        
        plotter.add_axes()
        plotter.add_legend(loc='upper left')
        plotter.add_title("Реальная МРТ + виртуальная опухоль", 
                         font_size=16, color="white")
        plotter.show()


def main():
    # === ПУТИ К ФАЙЛАМ ===
    NRRD_PATH = "data_real/Segmentation.nrrd"  # ← УКАЖИ ПУТЬ
    LABELS_CSV = "data_real/Segmentation.labels.csv"  # ← УКАЖИ ПУТЬ
    OUTPUT_DIR = "data_real"
    
    # 1. Загрузка
    loader = RealBreastDataLoader(NRRD_PATH, LABELS_CSV)
    loader.load_labels()
    volume = loader.load_volume()
    segmentation = loader.load_segmentation(NRRD_PATH)
    
    # 2. Создание физических карт
    eps_map, cond_map, temp_map, breast_mask, tissue_type_map = loader.create_tissue_maps()
    
    # 3. Добавление виртуальной опухоли
    temp_map, tumor_pos = loader.add_virtual_tumor(
        temp_map, breast_mask, tissue_type_map,
        tumor_radius_voxels=15,
        tumor_temp_increase=3.0
    )
    
    # 4. Сохранение для расчетов
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "real_eps_map.npy"), eps_map)
    np.save(os.path.join(OUTPUT_DIR, "real_cond_map.npy"), cond_map)
    np.save(os.path.join(OUTPUT_DIR, "real_temp_map.npy"), temp_map)
    np.save(os.path.join(OUTPUT_DIR, "real_breast_mask.npy"), breast_mask)
    np.save(os.path.join(OUTPUT_DIR, "real_tissue_type_map.npy"), tissue_type_map)
    
    # Сохраняем метаданные
    metadata = {
        'spacing': loader.spacing,
        'shape': volume.shape,
        'tumor_pos': tumor_pos,
    }
    np.save(os.path.join(OUTPUT_DIR, "real_metadata.npy"), metadata)
    
    print(f"\n💾 Все файлы сохранены в {OUTPUT_DIR}/")
    
    # 5. Визуализация
    loader.visualize(temp_map, breast_mask, tumor_pos)


if __name__ == "__main__":
    main()