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
        print("📋 Загрузка меток тканей...")
        try:
            df = pd.read_csv(self.labels_csv_path)
        except Exception as e:
            print(f"   ❌ Ошибка чтения CSV: {e}")
            self.label_map = {1: 'fat', 2: 'clot', 3: 'skin'}
            return self.label_map

        print(f"   Найдено строк: {len(df)}")
        print(f"   Колонки: {list(df.columns)}")

        id_col = None
        for col in df.columns:
            if any(k in str(col).lower() for k in ['label', 'id', 'value']):
                id_col = col
                break

        name_col = None
        for col in df.columns:
            if any(k in str(col).lower() for k in ['name', 'tissue', 'segment']):
                name_col = col
                break

        if id_col is None:
            id_col = df.columns[0]
        if name_col is None:
            name_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        print(f"   ✅ ID='{id_col}', Name='{name_col}'")

        for _, row in df.iterrows():
            try:
                label_id = int(row[id_col])
                name = str(row[name_col]).lower().strip()
                if label_id == 0 or not name or name == 'background':
                    continue
                self.label_map[label_id] = name
                print(f"   {label_id}: {name}")
            except (ValueError, TypeError):
                pass

        if not self.label_map:
            self.label_map = {1: 'fat', 2: 'clot', 3: 'skin'}

        print(f"\n✅ Итоговый словарь: {self.label_map}")
        return self.label_map

    def load_volume(self):
        print(f"\n📂 Загрузка МРТ: {self.nrrd_path}")
        image = sitk.ReadImage(self.nrrd_path)
        self.volume = sitk.GetArrayFromImage(image).astype(np.float32)
        self.spacing = image.GetSpacing()
        print(f"   Размер: {self.volume.shape} (Z, Y, X)")
        print(f"   Spacing: {self.spacing} мм")
        return self.volume

    def load_segmentation(self, seg_nrrd_path):
        print(f"\n🎯 Загрузка сегментации: {seg_nrrd_path}")
        seg_image = sitk.ReadImage(seg_nrrd_path)
        self.segmentation = sitk.GetArrayFromImage(seg_image).astype(np.int32)
        print(f"   Размер: {self.segmentation.shape}")
        print(f"   Уникальные метки: {np.unique(self.segmentation)}")

        for label_id, name in self.label_map.items():
            count = np.sum(self.segmentation == label_id)
            if count > 0:
                volume_ml = count * np.prod(self.spacing) / 1000
                print(f"   {name}: {count:,} вокселей ({volume_ml:.1f} мл)")

        return self.segmentation
            
    def create_tissue_maps(self):
        """
        Создает карты физических свойств на основе сегментации.
        ИСПРАВЛЕНАЯ ВЕРСИЯ:
        1. Железа = жир ВНУТРИ груди (не подкожный), если отдельной метки нет
        2. Увеличен температурный контраст между тканями
        3. Расширен диапазон клиппинга (30-42°C)
        4. Послойное сглаживание для сохранения тонких структур
        """
        print("\n🔬 Генерация физических свойств тканей...")
        shape = self.segmentation.shape
        
        eps_map = np.ones(shape, dtype=np.float32)       # Воздух = 1.0
        cond_map = np.zeros(shape, dtype=np.float32)      # Воздух = 0
        temp_map = np.ones(shape, dtype=np.float32) * 20.0  # Воздух = 20°C
        tissue_type_map = np.zeros(shape, dtype=np.int32)
        
        breast_mask = self.segmentation > 0
        
        # ✅ ИСПРАВЛЕНИЕ 1: УВЕЛИЧЕННЫЙ КОНТРАСТ ТЕМПЕРАТУР
        # Разница между тканями должна быть > 1°C, чтобы выжить после сглаживания
        tissue_props = {
            'fat':  {'eps': 5.0,  'std_eps': 0.5, 'cond': 0.10, 'std_cond': 0.03,
                    'temp': 35.0, 'type_id': 1},
            'clot': {'eps': 55.0, 'std_eps': 5.0, 'cond': 3.5,  'std_cond': 0.5,
                    'temp': 38.5, 'type_id': 2},   # ← Кровь ГОРЯЧАЯ (было 37.0)
            'skin': {'eps': 35.0, 'std_eps': 4.0, 'cond': 1.0,  'std_cond': 0.2,
                    'temp': 32.0, 'type_id': 3},   # ← Кожа ХОЛОДНАЯ (было 33.8)
        }
        
        # Заполнение тканей из сегментации
        for label_id, name in self.label_map.items():
            if label_id == 0:
                continue
            
            name_lower = name.lower()
            props = None
            for tissue_name, tissue_data in tissue_props.items():
                if tissue_name in name_lower:
                    props = tissue_data
                    break
            
            if props is None:
                print(f"   ⚠️ Неизвестная ткань: {name}, пропускаем")
                continue
            
            mask = self.segmentation == label_id
            count = np.sum(mask)
            if count == 0:
                continue
            
            eps_map[mask] = np.random.normal(props['eps'], props['std_eps'], count)
            cond_map[mask] = np.random.normal(props['cond'], props['std_cond'], count)
            temp_map[mask] = props['temp']
            tissue_type_map[mask] = props['type_id']
            
            print(f"   ✅ {name}: ε={props['eps']:.1f}, σ={props['cond']:.2f}, "
                f"T={props['temp']:.1f}°C ({count:,} вокселей)")
        
        # ✅ ИСПРАВЛЕНИЕ 2: ЖЕЛЕЗИСТАЯ ТКАНЬ
        # Если в сегментации нет отдельной метки "gland", 
        # считаем железой внутренний жир (не у поверхности кожи)
        gland_mask_auto = np.zeros(shape, dtype=bool)
        
        has_gland_label = any('gland' in name.lower() for name in self.label_map.values())
        
        if not has_gland_label:
            print("   ℹ️ Метки 'gland' нет в сегментации. Создаём автоматически...")
            
            from scipy.ndimage import distance_transform_edt
            
            # Железа = ткань внутри груди, которая НЕ является кожей и НЕ является кровью
            # И находится глубже 3 мм от поверхности (чтобы отделить от подкожного жира)
            dist_from_surface = distance_transform_edt(~breast_mask).astype(np.float32)
            spacing_z = self.spacing[2] if hasattr(self, 'spacing') and self.spacing else 3.0
            min_depth_mm = 3.0
            min_depth_voxels = min_depth_mm / spacing_z
            
            inner_mask = breast_mask & (dist_from_surface > min_depth_voxels)
            non_skin_non_clot = (tissue_type_map != 3) & (tissue_type_map != 2)
            gland_mask_auto = inner_mask & non_skin_non_clot
            
            gland_count = np.sum(gland_mask_auto)
            if gland_count > 0:
                eps_map[gland_mask_auto] = np.random.normal(45.0, 5.0, gland_count)
                cond_map[gland_mask_auto] = np.random.normal(2.4, 0.4, gland_count)
                temp_map[gland_mask_auto] = 36.0  # ← Железа теплее жира (35.0)
                tissue_type_map[gland_mask_auto] = 4
                print(f"   ✅ Железистая ткань (авто): {gland_count:,} вокселей, T=36.0°C")
            else:
                print("   ❌ Не удалось создать железистую ткань автоматически!")
        else:
            print("   ℹ️ Метка 'gland' найдена в сегментации.")
        
        # ✅ ИСПРАВЛЕНИЕ 3: ПОСЛОЙНОЕ СГЛАЖИВАНИЕ
        # Каждая ткань сглаживается отдельно, чтобы не смешиваться с соседними
        from scipy.ndimage import gaussian_filter
        
        print("   🎨 Послойное сглаживание (сохранение контраста)...")
        smoothed_temp = temp_map.copy()
        smoothed_eps = eps_map.copy()
        smoothed_cond = cond_map.copy()
        
        for tid in [1, 2, 3, 4]:
            tmask = (tissue_type_map == tid)
            if np.sum(tmask) == 0:
                continue
            
            # Кожа и кровь — минимальное сглаживание (тонкие структуры)
            # Жир и железа — стандартное сглаживание
            sigma = 0.3 if tid in [2, 3] else 0.8
            
            t_temp = np.where(tmask, temp_map, 0)
            t_eps = np.where(tmask, eps_map, 0)
            t_cond = np.where(tmask, cond_map, 0)
            
            s_temp = gaussian_filter(t_temp, sigma=sigma)
            s_eps = gaussian_filter(t_eps, sigma=sigma)
            s_cond = gaussian_filter(t_cond, sigma=sigma)
            
            smoothed_temp[tmask] = s_temp[tmask]
            smoothed_eps[tmask] = s_eps[tmask]
            smoothed_cond[tmask] = s_cond[tmask]
        
        temp_map = smoothed_temp
        eps_map = smoothed_eps
        cond_map = smoothed_cond
        
        # ✅ ИСПРАВЛЕНИЕ 4: РАСШИРЕННЫЙ ДИАПАЗОН КЛИППИНГА
        # Было [34.0, 39.5] — обрезало кожу (32°C) и воздух (20°C)
        temp_map = np.clip(temp_map, 30.0, 42.0)
        temp_map[~breast_mask] = 20.0
        eps_map[~breast_mask] = 1.0
        cond_map[~breast_mask] = 0.0
        
        # Диагностика ПОСЛЕ обработки
        print(f"\n📊 Температура по тканям (ПОСЛЕ обработки):")
        tissue_names = {1: 'Жир (fat)', 2: 'Кровь (clot)', 3: 'Кожа (skin)', 4: 'Железа'}
        for tid, name in tissue_names.items():
            tmask = (tissue_type_map == tid)
            if np.sum(tmask) > 0:
                t_mean = temp_map[tmask].mean()
                t_min = temp_map[tmask].min()
                t_max = temp_map[tmask].max()
                print(f"   {name:15s}: {t_mean:5.2f}°C "
                    f"(min={t_min:.2f}, max={t_max:.2f}, n={np.sum(tmask):,})")
            else:
                print(f"   {name:15s}: НЕТ ВОКСЕЛЕЙ")
        
        # Проверка различимости
        fat_t = temp_map[tissue_type_map == 1].mean() if np.any(tissue_type_map == 1) else None
        clot_t = temp_map[tissue_type_map == 2].mean() if np.any(tissue_type_map == 2) else None
        skin_t = temp_map[tissue_type_map == 3].mean() if np.any(tissue_type_map == 3) else None
        gland_t = temp_map[tissue_type_map == 4].mean() if np.any(tissue_type_map == 4) else None
        
        print(f"\n🔍 Разницы температур:")
        if fat_t and clot_t:
            d = clot_t - fat_t
            status = "✅" if abs(d) > 0.5 else "⚠️"
            print(f"   {status} clot-fat:  {d:+.2f}°C")
        if fat_t and skin_t:
            d = skin_t - fat_t
            status = "✅" if abs(d) > 0.5 else "⚠️"
            print(f"   {status} skin-fat:  {d:+.2f}°C")
        if fat_t and gland_t:
            d = gland_t - fat_t
            status = "✅" if abs(d) > 0.5 else "⚠️"
            print(f"   {status} gland-fat: {d:+.2f}°C")
        
        print(f"\n✅ Карты созданы:")
        print(f"   breast_mask: {np.sum(breast_mask):,} вокселей")
        print(f"   T диапазон: [{temp_map[breast_mask].min():.2f}, "
            f"{temp_map[breast_mask].max():.2f}]°C")
        
        return eps_map, cond_map, temp_map, breast_mask, tissue_type_map

    def add_virtual_tumor(self, temp_map, breast_mask, tissue_type_map,
                         tumor_radius_voxels=15, tumor_temp_increase=3.0):
        print(f"\n🎯 Добавление виртуальной опухоли (R={tumor_radius_voxels} вокселей)...")

        valid_mask = (tissue_type_map == 4) | (tissue_type_map == 1)
        valid_z, valid_y, valid_x = np.where(valid_mask)

        if len(valid_z) == 0:
            print("   ❌ Не найдено подходящих позиций!")
            return temp_map, None

        idx = np.random.randint(0, len(valid_z))
        tumor_z, tumor_y, tumor_x = valid_z[idx], valid_y[idx], valid_x[idx]
        print(f"   Позиция: Z={tumor_z}, Y={tumor_y}, X={tumor_x}")

        d, h, w = temp_map.shape
        z, y, x = np.ogrid[:d, :h, :w]
        dist_to_tumor = np.sqrt((z - tumor_z)**2 + (y - tumor_y)**2 + (x - tumor_x)**2)

        tumor_sigma = tumor_radius_voxels * 1.5
        temp_map += tumor_temp_increase * np.exp(-dist_to_tumor**2 / (2 * tumor_sigma**2)) * breast_mask

        inflammation_radius = tumor_radius_voxels * 2.5
        temp_map += 0.8 * np.exp(-dist_to_tumor**2 / (2 * inflammation_radius**2)) * breast_mask

        # ✅ Расширенный диапазон клиппинга
        temp_map = np.clip(temp_map, 30.0, 42.0)

        print(f"   ✅ Опухоль добавлена: +{tumor_temp_increase:.1f}°C")
        print(f"   Макс T после: {temp_map[breast_mask].max():.2f}°C")

        return temp_map, (tumor_z, tumor_y, tumor_x)

    def visualize(self, temp_map, breast_mask, tumor_pos=None):
        print("\n🎨 Визуализация...")

        temp_grid = temp_map.transpose(2, 1, 0)
        breast_grid = breast_mask.transpose(2, 1, 0).astype(np.uint8)

        grid = pv.ImageData(dimensions=temp_grid.shape)
        grid.point_data["temperature"] = temp_grid.flatten(order="F")
        grid.point_data["breast"] = breast_grid.flatten(order="F")

        plotter = pv.Plotter(window_size=[1400, 900])
        plotter.set_background("black")

        breast_surf = grid.contour(isosurfaces=[0.5], scalars="breast")
        plotter.add_mesh(breast_surf, color="peachpuff", opacity=0.3, label="Breast")

        # ✅ Расширенный clim для визуализации кожи и clot
        clim = [30.0, 42.0]

        try:
            hot_zone = grid.threshold(value=37.5, scalars="temperature")
            if hot_zone.n_points > 0:
                hot_surf = hot_zone.contour(isosurfaces=[37.5], scalars="temperature")
                plotter.add_mesh(hot_surf, color="red", opacity=0.7, label="Hot zone (>37.5)")
        except:
            pass

        bounds = grid.bounds
        z_mid = (bounds[4] + bounds[5]) / 2
        y_mid = (bounds[2] + bounds[3]) / 2

        slice_z = grid.slice(normal='z', origin=(0, 0, z_mid))
        plotter.add_mesh(slice_z, scalars="temperature", cmap="jet",
                        clim=clim, opacity=0.9)

        slice_y = grid.slice(normal='y', origin=(0, y_mid, 0))
        plotter.add_mesh(slice_y, scalars="temperature", cmap="jet",
                        clim=clim, opacity=0.9)

        if tumor_pos:
            tz, ty, tx = tumor_pos
            tumor_point = np.array([[tx, ty, tz]], dtype=np.float32)
            plotter.add_points(tumor_point, color="yellow", point_size=20,
                             render_points_as_spheres=True, label="Tumor")

        plotter.add_axes()
        plotter.add_legend(loc='upper left')
        plotter.add_title("Real MRI + Virtual Tumor", font_size=16, color="white")
        plotter.show()


def main():
    NRRD_PATH = "data_real/Segmentation.nrrd"
    LABELS_CSV = "data_real/Segmentation.labels.csv"
    OUTPUT_DIR = "data_real"

    loader = RealBreastDataLoader(NRRD_PATH, LABELS_CSV)
    loader.load_labels()
    volume = loader.load_volume()
    segmentation = loader.load_segmentation(NRRD_PATH)

    eps_map, cond_map, temp_map, breast_mask, tissue_type_map = loader.create_tissue_maps()

    temp_map, tumor_pos = loader.add_virtual_tumor(
        temp_map, breast_mask, tissue_type_map,
        tumor_radius_voxels=15,
        tumor_temp_increase=3.0
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "real_eps_map.npy"), eps_map)
    np.save(os.path.join(OUTPUT_DIR, "real_cond_map.npy"), cond_map)
    np.save(os.path.join(OUTPUT_DIR, "real_temp_map.npy"), temp_map)
    np.save(os.path.join(OUTPUT_DIR, "real_breast_mask.npy"), breast_mask)
    np.save(os.path.join(OUTPUT_DIR, "real_tissue_type_map.npy"), tissue_type_map)

    metadata = {
        'spacing': loader.spacing,
        'shape': volume.shape,
        'tumor_pos': tumor_pos,
    }
    np.save(os.path.join(OUTPUT_DIR, "real_metadata.npy"), metadata)

    print(f"\n💾 Все файлы сохранены в {OUTPUT_DIR}/")

    loader.visualize(temp_map, breast_mask, tumor_pos)


if __name__ == "__main__":
    main()