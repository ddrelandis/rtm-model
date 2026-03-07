import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, distance_transform_edt
import time

# =============================================================================
# 🏗️ КЛАСС МОДЕЛИ
# =============================================================================

class BreastRadiometryModelReal:
    def __init__(self, freq_ghz=3.5, resolution_mm=2, birads_category='B', temp_vmin=None, temp_vmax=None):  # 🔥 Новые параметры
        self.freq = freq_ghz * 1e9
        self.c = 3e8
        self.lambda0 = self.c / self.freq
        self.res = resolution_mm / 1000.0
        self.tumor_center = None
        self.birads_category = birads_category
        
        self.birads_density = {
            'A': (0.10, 0.25),
            'B': (0.26, 0.50),
            'C': (0.51, 0.75),
            'D': (0.76, 0.90)
        }

        self.temp_vmin = temp_vmin  # Ручная мин. температура
        self.temp_vmax = temp_vmax  # Ручная макс. температура
        
        # 🔥 Значения по умолчанию (физиологический диапазон)
        if self.temp_vmin is None:
            self.temp_vmin = 33.0
        if self.temp_vmax is None:
            self.temp_vmax = 40.0

        self.tissue_props = {
            'fat': {
                'mean_eps': 10.5, 'std_eps': 1.5, 
                'mean_cond': 0.15, 'std_cond': 0.05, 
                'temp_base': 35.0, 'temp_offset': 0.0
            },
            'fat_subcutaneous': {
                'mean_eps': 9.5, 'std_eps': 1.2, 
                'mean_cond': 0.12, 'std_cond': 0.04, 
                'temp_base': 34.8, 'temp_offset': 0.0
            },
            'fat_retromammary': {
                'mean_eps': 10.0, 'std_eps': 1.3, 
                'mean_cond': 0.14, 'std_cond': 0.04, 
                'temp_base': 35.0, 'temp_offset': 0.0
            },
            'gland': {
                'mean_eps': 48.0, 'std_eps': 6.0, 
                'mean_cond': 2.6, 'std_cond': 0.4, 
                'temp_base': 35.0, 'temp_offset': 0.8
            },
            'gland_ducts': {
                'mean_eps': 52.0, 'std_eps': 5.0, 
                'mean_cond': 3.0, 'std_cond': 0.5, 
                'temp_base': 35.0, 'temp_offset': 1.0
            },
            'connective': {
                'mean_eps': 35.0, 'std_eps': 4.0, 
                'mean_cond': 1.5, 'std_cond': 0.3, 
                'temp_base': 35.0, 'temp_offset': 0.3
            },
            'tumor': {
                'mean_eps': 58.0, 'std_eps': 8.0, 
                'mean_cond': 4.2, 'std_cond': 0.8, 
                'temp_base': 38.0, 'temp_offset': 0.0
            },
            'nipple': {
                'mean_eps': 52.0, 'std_eps': 5.0, 
                'mean_cond': 3.0, 'std_cond': 0.5, 
                'temp_base': 35.0, 'temp_offset': 0.6
            },
            'body': {
                'mean_eps': 50.0, 'std_eps': 5.0, 
                'mean_cond': 2.0, 'std_cond': 0.3, 
                'temp_base': 35.0, 'temp_offset': 0.0
            },
            'skin': {
                'mean_eps': 38.0, 'std_eps': 4.0, 
                'mean_cond': 1.2, 'std_cond': 0.2, 
                'temp_base': 33.8, 'temp_offset': 0.0
            }
        }

    def get_tissue_values(self, tissue_type, shape):
        props = self.tissue_props[tissue_type]
        eps_map = np.clip(np.random.normal(props['mean_eps'], props['std_eps'], shape), 1.0, None)
        cond_map = np.clip(np.random.normal(props['mean_cond'], props['std_cond'], shape), 0.01, None)
        temp_map = np.ones(shape) * props['temp_base']
        return eps_map, cond_map, temp_map, props['temp_offset']

    def create_anatomical_phantom(self, shape=(160, 200), tumor_radius=12, tumor_pos=None):
        """
        Создание фантома с ВЫСОКИМ РАЗРЕШЕНИЕМ.
        shape: (height, width) в пикселях
        tumor_radius: масштабированное значение (пропорционально разрешению)
        """
        start_time = time.time()
        
        h, w = shape
        eps_map = np.zeros(shape)
        cond_map = np.zeros(shape)
        temp_map = np.zeros(shape)
        temp_offset_map = np.zeros(shape)
        tissue_type_map = np.zeros(shape, dtype=int)
        
        y, x = np.ogrid[:h, :w]
        center_x = w / 2.0
        
        # 🔥 МАСШТАБИРУЕМЫЕ ПАРАМЕТРЫ (пропорционально разрешению)
        scale_factor = h / 80.0  # Базовое разрешение было 80
        
        print(f"🔍 Разрешение: {w}×{h} пикселей (масштаб: {scale_factor:.2f}×)")
        
        # 1. ФОРМА ГРУДИ
        top_y = int(h * 0.12)
        breast_width_top = w * 0.06
        breast_width_mid = w * 0.35
        breast_width_base = w * 0.55
        
        breast_mask = np.zeros(shape, dtype=bool)
        
        for yi in range(top_y, h):
            normalized_y = (yi - top_y) / (h - top_y)
            if normalized_y < 0.25:
                width_factor = breast_width_top + (breast_width_mid - breast_width_top) * (normalized_y / 0.25) ** 0.5
            elif normalized_y < 0.6:
                width_factor = breast_width_mid + (breast_width_base - breast_width_mid) * ((normalized_y - 0.25) / 0.35)
            else:
                width_factor = breast_width_base
            
            x_left = int(center_x - width_factor)
            x_right = int(center_x + width_factor)
            x_left = max(0, x_left)
            x_right = min(w, x_right)
            breast_mask[yi, x_left:x_right] = True
        
        # 🔥 Масштабируем сглаживание формы
        breast_mask = binary_dilation(breast_mask, iterations=max(1, int(2 * scale_factor)))
        breast_mask = gaussian_filter(breast_mask.astype(float), sigma=0.8 * scale_factor) > 0.5
        
        # 2. СОСОК
        nipple_center_y = int(h * 0.15)
        nipple_center_x = int(w / 2.0)
        areola_radius = int(w * 0.10)
        nipple_radius = int(w * 0.04)
        
        areola_mask = (x - nipple_center_x)**2 + (y - nipple_center_y)**2 <= areola_radius**2
        areola_mask = areola_mask & breast_mask
        
        nipple_mask = (x - nipple_center_x)**2 + (y - nipple_center_y)**2 <= nipple_radius**2
        nipple_mask = nipple_mask & areola_mask
        
        # 3. СЛОИ
        skin_thickness = max(2, int(2 * scale_factor))
        breast_boundary = binary_erosion(breast_mask, iterations=skin_thickness) ^ breast_mask
        skin_mask = breast_boundary & breast_mask
        
        subcut_thickness = max(4, int(h * 0.06))
        subcutaneous_mask = binary_erosion(breast_mask, iterations=skin_thickness)
        subcutaneous_mask = binary_erosion(subcutaneous_mask, iterations=subcut_thickness) ^ binary_erosion(breast_mask, iterations=skin_thickness)
        subcutaneous_mask = subcutaneous_mask & breast_mask & ~skin_mask
        
        retro_y_start = int(h * 0.65)
        retromammary_mask = (y >= retro_y_start) & breast_mask & ~skin_mask & ~subcutaneous_mask
        
        glandular_mask = breast_mask & ~skin_mask & ~subcutaneous_mask & ~retromammary_mask & ~areola_mask
        
        body_transition_y = int(h * 0.75)
        body_mask = (y >= body_transition_y) & breast_mask
        
        # 4. НЕОДНОРОДНАЯ СТРУКТУРА
        density_range = self.birads_density[self.birads_category]
        target_gland_fraction = np.random.uniform(density_range[0], density_range[1])
        
        print(f"📊 BI-RADS категория: {self.birads_category}")
        print(f"   Целевая доля железистой ткани: {target_gland_fraction*100:.1f}%")
        
        # 🔥 Масштабируем количество структур
        n_lobes = np.random.randint(15, 21)
        lobe_mask = np.zeros(shape, dtype=bool)
        gland_center_y = int(h * 0.45)
        gland_center_x = int(w / 2.0)
        
        for i in range(n_lobes):
            angle = (2 * np.pi * i) / n_lobes
            lobe_width = np.random.uniform(0.15, 0.25) * np.pi / n_lobes
            dy = y - gland_center_y
            dx = x - gland_center_x
            angle_map = np.arctan2(dy, dx)
            angle_diff = np.abs(angle_map - angle)
            angle_diff = np.minimum(angle_diff, 2*np.pi - angle_diff)
            sector_mask = (angle_diff < lobe_width) & glandular_mask
            dist_from_center = np.sqrt(dx**2 + dy**2)
            sector_mask = sector_mask & (dist_from_center < w * 0.4)
            lobe_mask = lobe_mask | sector_mask
        
        # 🔥 Масштабируем количество долек
        n_lobules = int(np.sum(lobe_mask) * 0.003)
        lobule_mask = np.zeros(shape, dtype=bool)
        lobule_indices = np.where(lobe_mask)
        for _ in range(n_lobules):
            if len(lobule_indices[0]) == 0:
                break
            idx = np.random.randint(0, len(lobule_indices[0]))
            cy, cx = lobule_indices[0][idx], lobule_indices[1][idx]
            lobule_size = max(4, int(np.random.randint(4, 10) * scale_factor))
            yy, xx = np.ogrid[:h, :w]
            lobule_region = (xx - cx)**2 + (yy - cy)**2 <= lobule_size**2
            lobule_mask = lobule_mask | (lobule_region & lobe_mask)
        
        # 🔥 Масштабируем протоки
        n_ducts = min(n_lobes, 12)
        duct_mask = np.zeros(shape, dtype=bool)
        for i in range(n_ducts):
            angle = (2 * np.pi * i) / n_ducts + np.random.uniform(-0.2, 0.2)
            t = np.linspace(0, 1, 100)
            for ti in t:
                px = int(nipple_center_x + ti * (w * 0.35) * np.cos(angle))
                py = int(nipple_center_y + ti * (h * 0.4) * np.sin(angle))
                if 0 <= px < w and 0 <= py < h:
                    duct_radius = max(3, int(3 * scale_factor))
                    duct_region = (x - px)**2 + (y - py)**2 <= duct_radius**2
                    duct_mask = duct_mask | (duct_region & glandular_mask)
        
        # 🔥 Масштабируем соединительную ткань
        connective_mask = np.zeros(shape, dtype=bool)
        n_fibers = max(60, int(60 * scale_factor))
        for _ in range(n_fibers):
            fy = np.random.randint(top_y, h)
            fx = np.random.randint(int(center_x - breast_width_base), int(center_x + breast_width_base))
            fiber_length = max(10, int(np.random.randint(10, 25) * scale_factor))
            fiber_angle = np.random.uniform(0, 2*np.pi)
            for l in range(fiber_length):
                px = int(fx + l * np.cos(fiber_angle))
                py = int(fy + l * np.sin(fiber_angle))
                if 0 <= px < w and 0 <= py < h:
                    fiber_radius = max(2, int(2 * scale_factor))
                    fiber_region = (x - px)**2 + (y - py)**2 <= fiber_radius**2
                    connective_mask = connective_mask | (fiber_region & glandular_mask)
        
        available_gland_area = np.sum(glandular_mask)
        target_gland_area = int(available_gland_area * target_gland_fraction)
        
        gland_priority = np.zeros(shape)
        gland_priority[lobule_mask] = 3
        gland_priority[duct_mask] = 2
        gland_priority[lobe_mask] = 1
        
        gland_indices = np.where(glandular_mask & (gland_priority > 0))
        if len(gland_indices[0]) > 0:
            priorities = gland_priority[gland_indices]
            sorted_idx = np.argsort(-priorities)
            gland_filled = 0
            final_gland_mask = np.zeros(shape, dtype=bool)
            for idx in sorted_idx:
                if gland_filled >= target_gland_area:
                    break
                gy, gx = gland_indices[0][idx], gland_indices[1][idx]
                final_gland_mask[gy, gx] = True
                gland_filled += 1
            intragland_fat_mask = glandular_mask & ~final_gland_mask
        else:
            final_gland_mask = np.zeros(shape, dtype=bool)
            intragland_fat_mask = glandular_mask
        
        # 5. ЗАПОЛНЕНИЕ ТКАНЯМИ
        tissue_type_map = np.zeros(shape, dtype=int)
        
        if np.any(subcutaneous_mask):
            e, c, t, offset = self.get_tissue_values('fat_subcutaneous', np.sum(subcutaneous_mask))
            eps_map[subcutaneous_mask] = e
            cond_map[subcutaneous_mask] = c
            temp_map[subcutaneous_mask] = t
            temp_offset_map[subcutaneous_mask] = offset
            tissue_type_map[subcutaneous_mask] = 1
        
        if np.any(final_gland_mask):
            e, c, t, offset = self.get_tissue_values('gland', np.sum(final_gland_mask))
            eps_map[final_gland_mask] = e
            cond_map[final_gland_mask] = c
            temp_map[final_gland_mask] = t
            temp_offset_map[final_gland_mask] = offset
            tissue_type_map[final_gland_mask] = 2
        
        if np.any(intragland_fat_mask):
            e, c, t, offset = self.get_tissue_values('fat', np.sum(intragland_fat_mask))
            eps_map[intragland_fat_mask] = e
            cond_map[intragland_fat_mask] = c
            temp_map[intragland_fat_mask] = t
            temp_offset_map[intragland_fat_mask] = offset
            tissue_type_map[intragland_fat_mask] = 3
        
        if np.any(retromammary_mask):
            e, c, t, offset = self.get_tissue_values('fat_retromammary', np.sum(retromammary_mask))
            eps_map[retromammary_mask] = e
            cond_map[retromammary_mask] = c
            temp_map[retromammary_mask] = t
            temp_offset_map[retromammary_mask] = offset
            tissue_type_map[retromammary_mask] = 4
        
        if np.any(connective_mask):
            e, c, t, offset = self.get_tissue_values('connective', np.sum(connective_mask))
            eps_map[connective_mask] = e
            cond_map[connective_mask] = c
            temp_map[connective_mask] = t
            temp_offset_map[connective_mask] = offset
            tissue_type_map[connective_mask] = 5
        
        if np.any(duct_mask):
            e, c, t, offset = self.get_tissue_values('gland_ducts', np.sum(duct_mask))
            eps_map[duct_mask] = e
            cond_map[duct_mask] = c
            temp_map[duct_mask] = t
            temp_offset_map[duct_mask] = offset
            tissue_type_map[duct_mask] = 6
        
        if np.any(lobule_mask & final_gland_mask):
            temp_offset_map[lobule_mask & final_gland_mask] = 0.9
            tissue_type_map[lobule_mask & final_gland_mask] = 7
        
        eps_map[~breast_mask] = 1.0
        cond_map[~breast_mask] = 0.0
        temp_map[~breast_mask] = 20.0
        temp_offset_map[~breast_mask] = 0.0
        
        if np.any(areola_mask):
            e, c, t, offset = self.get_tissue_values('gland', np.sum(areola_mask))
            eps_map[areola_mask] = e
            cond_map[areola_mask] = c
            temp_map[areola_mask] = t
            temp_offset_map[areola_mask] = 0.5
            tissue_type_map[areola_mask] = 9
        
        if np.any(nipple_mask):
            e, c, t, offset = self.get_tissue_values('nipple', np.sum(nipple_mask))
            eps_map[nipple_mask] = e
            cond_map[nipple_mask] = c
            temp_map[nipple_mask] = t
            temp_offset_map[nipple_mask] = offset
            tissue_type_map[nipple_mask] = 8
        
        if np.any(skin_mask):
            e, c, t, offset = self.get_tissue_values('skin', np.sum(skin_mask))
            eps_map[skin_mask] = e
            cond_map[skin_mask] = c
            temp_map[skin_mask] = t
            temp_offset_map[skin_mask] = offset
            tissue_type_map[skin_mask] = 10
        
        if np.any(body_mask):
            e, c, t, offset = self.get_tissue_values('body', np.sum(body_mask))
            eps_map[body_mask] = e
            cond_map[body_mask] = c
            temp_map[body_mask] = t
            temp_offset_map[body_mask] = offset
            tissue_type_map[body_mask] = 11
        
        # 6. ТЕМПЕРАТУРНЫЙ ГРАДИЕНТ
        dist_from_surface = distance_transform_edt(~breast_mask)
        dist_from_surface = dist_from_surface.astype(float)
        dist_from_surface[~breast_mask] = 0
        
        max_dist = dist_from_surface[breast_mask].max()
        if max_dist > 0:
            normalized_depth = dist_from_surface / max_dist
        else:
            normalized_depth = np.zeros_like(dist_from_surface)
        
        depth_gradient = 2.0 * (normalized_depth ** 0.6)
        temp_map = temp_map + depth_gradient * breast_mask
        temp_map = temp_map + temp_offset_map * breast_mask
        
        noise = np.random.normal(0, 0.08, shape)
        temp_map = temp_map + noise * breast_mask
        
        # 🔥 Масштабируем сглаживание температуры
        temp_map = gaussian_filter(temp_map, sigma=max(0.8, 0.8 * scale_factor))
        temp_map[~breast_mask] = 20.0
        temp_map = np.clip(temp_map, 34.0, 39.5)
        temp_map[~breast_mask] = 20.0
        
        # 7. ОПУХОЛЬ
        self.tumor_center = None
        tumor_ty, tumor_tx = None, None
        
        if tumor_pos is not None:
            tumor_ty, tumor_tx = tumor_pos
            if 0 <= tumor_ty < h and 0 <= tumor_tx < w:
                if final_gland_mask[tumor_ty, tumor_tx]:
                    print(f"✅ Опухоль создана в заданной позиции: Y={tumor_ty}, X={tumor_tx}")
                else:
                    print(f"⚠️ Позиция ({tumor_ty}, {tumor_tx}) вне железистой ткани! Коррекция...")
                    y_coords, x_coords = np.where(final_gland_mask)
                    if len(y_coords) > 0:
                        dists = np.sqrt((y_coords - tumor_ty)**2 + (x_coords - tumor_tx)**2)
                        nearest_idx = np.argmin(dists)
                        tumor_ty, tumor_tx = y_coords[nearest_idx], x_coords[nearest_idx]
                        print(f"✅ Скорректированная позиция: Y={tumor_ty}, X={tumor_tx}")
                    else:
                        print("⚠️ Не удалось скорректировать позицию")
                        tumor_pos = None
            else:
                print(f"⚠️ Позиция ({tumor_ty}, {tumor_tx}) вне сетки! Генерация случайной...")
                tumor_pos = None
        
        if tumor_pos is None:
            valid_y, valid_x = np.where(final_gland_mask & (y > h*0.30) & (y < h*0.65))
            if len(valid_y) > 0:
                idx = np.random.randint(0, len(valid_y))
                tumor_ty, tumor_tx = valid_y[idx], valid_x[idx]
                print(f"🎲 Опухоль создана в случайной позиции: Y={tumor_ty}, X={tumor_tx}")
            else:
                print("⚠️ Не удалось найти позицию для опухоли")
                return eps_map, cond_map, temp_map, breast_mask, areola_mask, nipple_mask, body_mask, tissue_type_map
        
        tumor_y, tumor_x = np.ogrid[:h, :w]
        dist_from_tumor = np.sqrt((tumor_x - tumor_tx)**2 + (tumor_y - tumor_ty)**2)
        
        # 🔥 Масштабируем опухоль
        tumor_sigma = tumor_radius * 1.5
        tumor_temp_elevation = 2.5 * np.exp(-dist_from_tumor**2 / (2 * tumor_sigma**2))
        
        temp_map = temp_map + tumor_temp_elevation * breast_mask
        temp_map = np.clip(temp_map, 34.0, 39.5)
        temp_map = gaussian_filter(temp_map, sigma=max(1.0, 1.0 * scale_factor))
        temp_map[~breast_mask] = 20.0
        
        inflammation_radius = tumor_radius * 3
        inflammation_effect = 0.5 * np.exp(-dist_from_tumor**2 / (2 * inflammation_radius**2))
        temp_map = temp_map + inflammation_effect * breast_mask
        temp_map = np.clip(temp_map, 34.0, 39.5)
        
        tumor_eps_elevation = 15.0 * np.exp(-dist_from_tumor**2 / (2 * tumor_sigma**2))
        eps_map = eps_map + tumor_eps_elevation * breast_mask
        eps_map = gaussian_filter(eps_map, sigma=max(1.5, 1.5 * scale_factor))
        eps_map[~breast_mask] = 1.0
        
        self.tumor_center = (tumor_ty, tumor_tx)
        
        elapsed = time.time() - start_time
        print(f"⏱️ Время создания фантома: {elapsed:.2f} сек")

        return eps_map, cond_map, temp_map, breast_mask, areola_mask, nipple_mask, body_mask, tissue_type_map

    def compute_sensitivity_kernel(self, mask, ant_pos):
        h, w = mask.shape
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - ant_pos[1])**2 + (y - ant_pos[0])**2) * self.res
        delta_eff = 0.02
        sensitivity = np.exp(-dist / delta_eff) * mask
        sum_sens = np.sum(sensitivity)
        if sum_sens > 0:
            return sensitivity / sum_sens
        return sensitivity

    def compute_emissivity(self, eps_map):
        sqrt_eps = np.sqrt(np.maximum(eps_map, 1.0))
        gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
        return 1.0 - gamma**2

    def forward_scan(self, eps_map, cond_map, temp_map, mask, scan_positions):
        measurements = []
        emissivity_avg = []
        for pos in scan_positions:
            kernel = self.compute_sensitivity_kernel(mask, pos)
            emissivity = self.compute_emissivity(eps_map)
            emissivity_avg.append(np.sum(kernel * emissivity))
            Tb = np.sum(kernel * temp_map * emissivity)
            measurements.append(Tb)
        return np.array(measurements), np.array(emissivity_avg)

    def reconstruct_simple(self, measurements, emissivity_avg, scan_positions, shape, mask):
        recon_field = np.zeros(shape)
        weight_sum = np.zeros(shape)
        
        for i, pos in enumerate(scan_positions):
            kernel = self.compute_sensitivity_kernel(mask, pos)
            emissivity_corr = emissivity_avg[i] if emissivity_avg[i] > 0.1 else 0.5
            Tb_corrected = measurements[i] / emissivity_corr
            recon_field += kernel * Tb_corrected
            weight_sum += kernel
            
        weight_sum[weight_sum == 0] = 1.0
        recon_field /= weight_sum
        recon_field = gaussian_filter(recon_field, sigma=2.0)
        
        # 🔥 ОБРЕЗАНИЕ до физиологического диапазона!
        recon_field = np.clip(recon_field, self.temp_vmin - 5, self.temp_vmax + 5)
        
        valid_data = recon_field[mask]
        if len(valid_data) > 0:
            min_t, max_t = np.percentile(valid_data, [5, 95])
            if max_t > min_t:
                recon_field = 35.0 + (recon_field - min_t) / (max_t - min_t) * 4.0
        
        # 🔥 ФИНАЛЬНОЕ обрезание после нормализации
        recon_field = np.clip(recon_field, self.temp_vmin, self.temp_vmax)
        recon_field[~mask] = np.nan
        return recon_field
# =============================================================================
# 📊 ФУНКЦИИ ВИЗУАЛИЗАЦИИ (ВСЕ!)
# =============================================================================

def plot_main_results(temp_true, temp_recon, breast_mask, tumor_center=None, 
                      areola_mask=None, nipple_mask=None, body_mask=None,
                      temp_vmin=33.0, temp_vmax=40.0):  # 🔥 Ручные параметры
    """
    temp_vmin, temp_vmax: фиксированный диапазон для обоих графиков
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # --- ГРАФИК 1: Истинное распределение T ---
    temp_display = temp_true.copy()
    temp_display[~breast_mask] = np.nan
    
    # 🔥 Используем ФИКСИРОВАННЫЙ диапазон (не авто!)
    im1 = axes[0].imshow(temp_display, cmap='jet', 
                         vmin=temp_vmin, vmax=temp_vmax, 
                         interpolation='gaussian')
    axes[0].set_title(f'Истинное распределение T\n[{temp_vmin:.1f}°C — {temp_vmax:.1f}°C]', 
                     fontsize=12, fontweight='bold')
    
    if nipple_mask is not None:
        axes[0].contour(nipple_mask, colors='darkred', linewidths=3, alpha=0.9)
    if areola_mask is not None:
        axes[0].contour(areola_mask, colors='coral', linewidths=2, alpha=0.7)
    
    if tumor_center:
        axes[0].plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, 
                    markeredgewidth=2, label='Опухоль')
        axes[0].legend(loc='lower right')
    
    cbar1 = plt.colorbar(im1, ax=axes[0], label='Температура (°C)')
    cbar1.set_ticks(np.linspace(temp_vmin, temp_vmax, 8))  # 🔥 8 делений
    
    # --- ГРАФИК 2: Реконструированное T ---
    temp_recon_display = temp_recon.copy()
    temp_recon_display[~breast_mask] = np.nan
    
    # 🔥 Тот же диапазон для сравнения!
    im2 = axes[1].imshow(temp_recon_display, cmap='jet', 
                         vmin=temp_vmin, vmax=temp_vmax, 
                         interpolation='gaussian')
    axes[1].set_title(f'Реконструированное T\n[{temp_vmin:.1f}°C — {temp_vmax:.1f}°C]', 
                     fontsize=12, fontweight='bold')
    
    if tumor_center:
        axes[1].plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, 
                    markeredgewidth=2)
    
    cbar2 = plt.colorbar(im2, ax=axes[1], label='Температура (°C)')
    cbar2.set_ticks(np.linspace(temp_vmin, temp_vmax, 8))
    
    # 🔥 Проверка на артефакты
    recon_valid = temp_recon[breast_mask]
    if np.any(recon_valid < temp_vmin) or np.any(recon_valid > temp_vmax):
        print(f"⚠️ ВНИМАНИЕ: Обнаружены значения вне диапазона!")
        print(f"   Мин: {np.nanmin(recon_valid):.2f}°C, Макс: {np.nanmax(recon_valid):.2f}°C")
    
    # --- ГРАФИК 3: Абсолютная ошибка ---
    diff = np.abs(temp_true - temp_recon)
    diff[~breast_mask] = np.nan
    vmin_err = 0
    vmax_err = min(3.0, np.nanmax(diff))  # 🔥 Максимум 3°C для ошибки
    
    im3 = axes[2].imshow(diff, cmap='magma', vmin=vmin_err, vmax=vmax_err, 
                         interpolation='gaussian')
    axes[2].set_title(f'Абсолютная ошибка\n[{vmin_err:.1f}°C — {vmax_err:.1f}°C]', 
                     fontsize=12, fontweight='bold')
    
    cbar3 = plt.colorbar(im3, ax=axes[2], label='Ошибка (°C)')
    cbar3.set_ticks(np.linspace(vmin_err, vmax_err, 5))
    
    plt.tight_layout()
    plt.savefig('01_main_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Диапазоны температур:")
    print(f"   Истинное T:     {np.min(temp_true[breast_mask]):.2f} — {np.max(temp_true[breast_mask]):.2f} °C")
    print(f"   Реконструированное: {np.nanmin(temp_recon[breast_mask]):.2f} — {np.nanmax(temp_recon[breast_mask]):.2f} °C")
    print(f"   Ошибка:         {vmin_err:.2f} — {vmax_err:.2f} °C")


def plot_tissue_composition(tissue_type_map, breast_mask, areola_mask, nipple_mask, body_mask, birads_category):
    """Визуализация неоднородной структуры тканей"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    tissue_names = {
        0: 'Фон', 1: 'Подкожный жир', 2: 'Железистая', 3: 'Внутрижелез. жир',
        4: 'Ретромаммарный', 5: 'Соединительная', 6: 'Протоки', 7: 'Дольки',
        8: 'Сосок', 9: 'Ареола', 10: 'Кожа', 11: 'Тело'
    }
    
    colors = [
        '#000000', '#F9F0D9', '#D4A5A5', '#F4E4C1', '#E8D5A3',
        '#E0C0C0', '#C48585', '#B87070', '#8B4513', '#CD853F', '#FFE4C4', '#A0A0A0'
    ]
    
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    im1 = axes[0].imshow(tissue_type_map, cmap=cmap, vmin=0, vmax=11)
    axes[0].set_title(f'Гистологическая структура (BI-RADS {birads_category})', fontsize=12, fontweight='bold')
    
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', label=name, 
                                   markerfacecolor=colors[i], markersize=10) 
                       for i, name in tissue_names.items() if i > 0]
    axes[0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), 
                   fontsize=8, framealpha=0.9)
    
    tissue_counts = {}
    total_pixels = np.sum(breast_mask)
    for i in range(1, 12):
        count = np.sum(tissue_type_map == i)
        if count > 0:
            tissue_counts[i] = count / total_pixels * 100
    
    labels = [tissue_names[i] for i in tissue_counts.keys()]
    sizes = list(tissue_counts.values())
    pie_colors = [colors[i] for i in tissue_counts.keys()]
    
    axes[1].pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', 
                startangle=90, textprops={'fontsize': 8})
    axes[1].set_title('Распределение тканей (%)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('07_tissue_composition.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_breast_anatomy(eps_map, breast_mask, areola_mask, nipple_mask, body_mask):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    eps_display = eps_map.copy()
    eps_display[~breast_mask] = np.nan
    vmin_eps = np.min(eps_map[breast_mask])
    vmax_eps = np.max(eps_map[breast_mask])
    
    im1 = axes[0].imshow(eps_display, cmap='viridis', vmin=vmin_eps, vmax=vmax_eps, interpolation='gaussian')
    axes[0].set_title(f'Диэлектрическая проницаемость\n[{vmin_eps:.1f} — {vmax_eps:.1f}]', fontsize=12, fontweight='bold')
    
    if nipple_mask is not None:
        axes[0].contour(nipple_mask, colors='darkred', linewidths=3, alpha=0.9)
    if areola_mask is not None:
        axes[0].contour(areola_mask, colors='coral', linewidths=2, alpha=0.7)
    
    cbar1 = plt.colorbar(im1, ax=axes[0], label='ε')
    cbar1.set_ticks(np.linspace(vmin_eps, vmax_eps, 6))
    
    anatomy = np.zeros(breast_mask.shape)
    anatomy[breast_mask] = 1
    anatomy[areola_mask] = 2
    anatomy[nipple_mask] = 3
    anatomy[body_mask] = 4
    
    im2 = axes[1].imshow(anatomy, cmap='tab10', vmin=0, vmax=4)
    axes[1].set_title('Анатомическая структура', fontsize=12, fontweight='bold')
    
    legend_text = '1 - Жировая ткань\n2 - Ареола\n3 - Сосок\n4 - Тело'
    axes[1].text(0.02, 0.95, legend_text, transform=axes[1].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.colorbar(im2, ax=axes[1], label='Тип ткани')
    
    plt.tight_layout()
    plt.savefig('08_breast_anatomy.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_temperature_gradient(temp_map, breast_mask, tumor_center=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h, w = temp_map.shape
    
    center_x = w // 2
    temp_profile = temp_map[:, center_x].copy()
    temp_profile[~breast_mask[:, center_x]] = np.nan
    
    y_coords = np.arange(h)
    valid_mask = breast_mask[:, center_x]
    
    valid_profile = temp_profile[valid_mask]
    if len(valid_profile) > 0:
        vmin_prof = np.nanmin(valid_profile)
        vmax_prof = np.nanmax(valid_profile)
    else:
        vmin_prof, vmax_prof = 33, 39
    
    axes[0].plot(valid_profile, y_coords[valid_mask], 'b-', linewidth=2.5)
    axes[0].fill_betweenx(y_coords[valid_mask], valid_profile, vmin_prof, alpha=0.3)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Температура (°C)', fontsize=11)
    axes[0].set_ylabel('Глубина (пиксели)', fontsize=11)
    axes[0].set_title(f'Температурный градиент\n[{vmin_prof:.1f}°C — {vmax_prof:.1f}°C]', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(34.5, color='green', linestyle='--', label='Поверхность (~34.5°C)')
    axes[0].axvline(37.0, color='red', linestyle='--', label='Грудная стенка (~37°C)')
    axes[0].legend(loc='lower right')
    
    valid_temps = temp_map[breast_mask]
    axes[1].hist(valid_temps, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(valid_temps.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {valid_temps.mean():.2f}°C')
    axes[1].axvline(valid_temps.min(), color='green', linestyle=':', linewidth=2, label=f'Мин: {valid_temps.min():.2f}°C')
    axes[1].axvline(valid_temps.max(), color='orange', linestyle=':', linewidth=2, label=f'Макс: {valid_temps.max():.2f}°C')
    axes[1].set_xlabel('Температура (°C)', fontsize=11)
    axes[1].set_ylabel('Количество пикселей', fontsize=11)
    axes[1].set_title(f'Распределение температур\n[{valid_temps.min():.1f}°C — {valid_temps.max():.1f}°C]', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('09_temperature_gradient.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_temperature_contours(temp_map, breast_mask, tumor_center=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    temp_display = temp_map.copy()
    temp_display[~breast_mask] = np.nan
    
    vmin_temp = np.nanmin(temp_display)
    vmax_temp = np.nanmax(temp_display)
    
    im = ax.imshow(temp_display, cmap='jet', vmin=vmin_temp, vmax=vmax_temp, interpolation='gaussian')
    
    contour_levels = np.arange(np.ceil(vmin_temp*10)/10, np.floor(vmax_temp*10)/10 + 0.1, 0.3)
    
    if len(contour_levels) > 1:
        cs = ax.contour(temp_display, levels=contour_levels, colors='white', linewidths=1.0, alpha=0.8)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f°C')
    
    if tumor_center:
        ax.plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, markeredgewidth=2, label='Опухоль')
        ax.legend(loc='lower right')
    
    ax.set_title(f'Изотермы температуры\n[{vmin_temp:.1f}°C — {vmax_temp:.1f}°C]', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax, label='Температура (°C)')
    cbar.set_ticks(np.linspace(vmin_temp, vmax_temp, 6))
    
    plt.tight_layout()
    plt.savefig('10_temperature_contours.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_temperature_difference_map(temp_map, tissue_type_map, breast_mask):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    tissue_temps = {}
    for i in range(1, 8):
        mask = tissue_type_map == i
        if np.sum(mask) > 100:
            tissue_temps[i] = np.mean(temp_map[mask])
    
    avg_temp = np.mean(temp_map[breast_mask])
    temp_diff = temp_map - avg_temp
    temp_diff[~breast_mask] = np.nan
    
    vmin_diff = np.nanmin(temp_diff)
    vmax_diff = np.nanmax(temp_diff)
    max_abs_diff = max(abs(vmin_diff), abs(vmax_diff))
    
    im1 = axes[0].imshow(temp_diff, cmap='RdBu_r', vmin=-max_abs_diff, vmax=max_abs_diff, interpolation='gaussian')
    axes[0].set_title(f'Отклонение от средней\n[{vmin_diff:.2f}°C — {vmax_diff:.2f}°C]', fontsize=12, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=axes[0], label='ΔT (°C)')
    cbar1.set_ticks(np.linspace(-max_abs_diff, max_abs_diff, 5))
    
    tissue_names = {1: 'Подкожный\nжир', 2: 'Железистая', 3: 'Внутрижелез.\nжир', 
                   4: 'Ретромаммарный', 5: 'Соединительная', 6: 'Протоки', 7: 'Дольки'}
    
    temps = [tissue_temps.get(i, 0) for i in range(1, 8)]
    names = [tissue_names.get(i, '') for i in range(1, 8)]
    
    axes[1].bar(range(len(temps)), temps, color=['#F9F0D9', '#D4A5A5', '#F4E4C1', 
                                                  '#E8D5A3', '#E0C0C0', '#C48585', '#B87070'])
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, fontsize=8)
    axes[1].axhline(avg_temp, color='red', linestyle='--', linewidth=2, label=f'Средняя: {avg_temp:.2f}°C')
    axes[1].set_ylabel('Температура (°C)', fontsize=11)
    axes[1].set_title('Температура по типам тканей', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].set_ylim(avg_temp - 2, avg_temp + 2)
    
    plt.tight_layout()
    plt.savefig('11_temperature_difference.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_sensitivity_kernels(model, breast_mask, scan_positions, n_show=5):
    fig, axes = plt.subplots(1, n_show, figsize=(20, 4))
    if n_show == 1:
        axes = [axes]
    
    step = max(1, len(scan_positions) // n_show)
    
    for i, ax in enumerate(axes):
        idx = i * step
        if idx >= len(scan_positions):
            idx = len(scan_positions) - 1
        pos = scan_positions[idx]
        kernel = model.compute_sensitivity_kernel(breast_mask, pos)
        
        im = ax.imshow(kernel, cmap='viridis', vmin=0, vmax=np.max(kernel)*1.2)
        ax.plot(pos[1], pos[0], 'r*', markersize=15, label='Антенна')
        ax.set_title(f'Антенна #{idx+1}\nПозиция: ({pos[0]}, {pos[1]})', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax, label='Норм. чувствительность')
    
    plt.suptitle('Функции чувствительности антенн', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('02_sensitivity_kernels.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_measurement_data(Tb_data, Tb_noisy, emissivity_avg, scan_positions):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x_antenna = [pos[1] for pos in scan_positions]
    
    axes[0].plot(x_antenna, Tb_data, 'bo-', linewidth=2, markersize=8, label='Без шума')
    axes[0].plot(x_antenna, Tb_noisy, 'rs--', linewidth=2, markersize=8, label='С шумом')
    axes[0].set_xlabel('Позиция антенны (X)', fontsize=11)
    axes[0].set_ylabel('Яркостная температура Tb (K)', fontsize=11)
    axes[0].set_title('Измерения яркостной температуры', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(x_antenna, emissivity_avg, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Позиция антенны (X)', fontsize=11)
    axes[1].set_ylabel('Средний коэффициент излучения', fontsize=11)
    axes[1].set_title('Коэффициент излучения по антеннам', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=emissivity_avg.mean(), color='r', linestyle='--', label=f'Среднее: {emissivity_avg.mean():.3f}')
    axes[1].legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('03_measurement_data.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_temperature_histogram(temp_true, temp_recon, breast_mask):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    true_vals = temp_true[breast_mask]
    recon_vals = temp_recon[breast_mask]
    
    axes[0].hist(true_vals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(true_vals.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {true_vals.mean():.2f}°C')
    axes[0].set_xlabel('Температура (°C)', fontsize=11)
    axes[0].set_ylabel('Количество пикселей', fontsize=11)
    axes[0].set_title(f'Распределение истинных температур\n[{true_vals.min():.1f}°C — {true_vals.max():.1f}°C]', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(recon_vals, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(recon_vals.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {recon_vals.mean():.2f}°C')
    axes[1].set_xlabel('Температура (°C)', fontsize=11)
    axes[1].set_ylabel('Количество пикселей', fontsize=11)
    axes[1].set_title(f'Распределение реконструированных температур\n[{recon_vals.min():.1f}°C — {recon_vals.max():.1f}°C]', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_temperature_histogram.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_cross_section(temp_true, temp_recon, breast_mask, tumor_center=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h, w = temp_true.shape
    
    if tumor_center:
        y_slice = int(tumor_center[0])
        x_range = np.arange(w)
        true_slice = temp_true[y_slice, :]
        recon_slice = temp_recon[y_slice, :]
        mask_slice = breast_mask[y_slice, :]
        
        slice_vals = np.concatenate([true_slice[mask_slice], recon_slice[mask_slice]])
        vmin_slice = np.nanmin(slice_vals)
        vmax_slice = np.nanmax(slice_vals)
        
        axes[0].plot(x_range[mask_slice], true_slice[mask_slice], 'b-', linewidth=2, label='Истинная T')
        axes[0].plot(x_range[mask_slice], recon_slice[mask_slice], 'r--', linewidth=2, label='Реконструированная T')
        axes[0].axvline(tumor_center[1], color='green', linestyle=':', linewidth=2, label='Центр опухоли')
        axes[0].set_xlabel('Позиция X (пиксели)', fontsize=11)
        axes[0].set_ylabel('Температура (°C)', fontsize=11)
        axes[0].set_title(f'Горизонтальный срез Y={y_slice}\n[{vmin_slice:.1f}°C — {vmax_slice:.1f}°C]', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(vmin_slice - 0.5, vmax_slice + 0.5)
        
        x_slice = int(tumor_center[1])
        y_range = np.arange(h)
        true_slice_v = temp_true[:, x_slice]
        recon_slice_v = temp_recon[:, x_slice]
        mask_slice_v = breast_mask[:, x_slice]
        
        slice_vals_v = np.concatenate([true_slice_v[mask_slice_v], recon_slice_v[mask_slice_v]])
        vmin_slice_v = np.nanmin(slice_vals_v)
        vmax_slice_v = np.nanmax(slice_vals_v)
        
        axes[1].plot(y_range[mask_slice_v], true_slice_v[mask_slice_v], 'b-', linewidth=2, label='Истинная T')
        axes[1].plot(y_range[mask_slice_v], recon_slice_v[mask_slice_v], 'r--', linewidth=2, label='Реконструированная T')
        axes[1].axvline(tumor_center[0], color='green', linestyle=':', linewidth=2, label='Центр опухоли')
        axes[1].set_xlabel('Позиция Y (пиксели)', fontsize=11)
        axes[1].set_ylabel('Температура (°C)', fontsize=11)
        axes[1].set_title(f'Вертикальный срез X={x_slice}\n[{vmin_slice_v:.1f}°C — {vmax_slice_v:.1f}°C]', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(vmin_slice_v - 0.5, vmax_slice_v + 0.5)
    else:
        axes[0].axis('off')
        axes[1].axis('off')
        axes[0].text(0.5, 0.5, 'Опухоль не найдена', ha='center', va='center', fontsize=14, transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'Опухоль не найдена', ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig('05_cross_section.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_emissivity_map(eps_map, breast_mask):
    sqrt_eps = np.sqrt(np.maximum(eps_map, 1.0))
    gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
    emissivity = 1.0 - gamma**2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    eps_inside = eps_map[breast_mask]
    vmin_eps = np.min(eps_inside)
    vmax_eps = np.max(eps_inside)
    
    im1 = axes[0].imshow(eps_map, cmap='viridis', vmin=vmin_eps, vmax=vmax_eps, interpolation='gaussian')
    axes[0].set_title(f'Диэлектрическая проницаемость\n[{vmin_eps:.1f} — {vmax_eps:.1f}]', fontsize=12, fontweight='bold')
    
    cbar1 = plt.colorbar(im1, ax=axes[0], label='ε')
    cbar1.set_ticks(np.linspace(vmin_eps, vmax_eps, 6))
    
    emissivity_display = emissivity.copy()
    emissivity_display[~breast_mask] = np.nan
    
    im2 = axes[1].imshow(emissivity_display, cmap='plasma', vmin=0.5, vmax=1.0, interpolation='gaussian')
    axes[1].set_title('Коэффициент излучения (Emissivity)\n[0.5 — 1.0]', fontsize=12, fontweight='bold')
    
    cbar2 = plt.colorbar(im2, ax=axes[1], label='Emissivity')
    cbar2.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    plt.tight_layout()
    plt.savefig('06_emissivity_map.png', dpi=150, bbox_inches='tight')
    plt.show()


def print_full_statistics(temp_true, temp_recon, breast_mask, Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, model, tissue_type_map=None):
    print("\n" + "="*70)
    print("📊 ПОЛНАЯ СТАТИСТИКА РАБОТЫ МОДЕЛИ")
    print("="*70)
    
    valid_true = temp_true[breast_mask]
    valid_recon = temp_recon[breast_mask]
    
    print("\n📍 ДИЭЛЕКТРИЧЕСКИЕ СВОЙСТВА (внутри груди):")
    print(f"   EPS:  {np.mean(eps_map[breast_mask]):6.2f} ± {np.std(eps_map[breast_mask]):.2f}")
    print(f"   COND: {np.mean(cond_map[breast_mask]):6.2f} ± {np.std(cond_map[breast_mask]):.2f} См/м")
    
    print("\n🌡️ ТЕМПЕРАТУРНАЯ СТАТИСТИКА:")
    print(f"   Истинная T:        {valid_true.mean():6.2f} ± {valid_true.std():.2f} °C")
    print(f"   Реконструированная: {valid_recon.mean():6.2f} ± {valid_recon.std():.2f} °C")
    print(f"   Смещение (bias):    {valid_recon.mean() - valid_true.mean():+.2f} °C")
    print(f"   Мин T:              {valid_true.min():6.2f} °C")
    print(f"   Макс T:             {valid_true.max():6.2f} °C")
    print(f"   Диапазон:           {valid_true.max() - valid_true.min():.2f} °C")
    
    temp_masked = temp_true.copy()
    temp_masked[~breast_mask] = 0
    grad_y, grad_x = np.gradient(temp_masked)
    grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
    grad_inside = grad_magnitude[breast_mask]
    print(f"   Плавность (средн. градиент): {grad_inside.mean():.4f} °C/пиксель")
    print(f"   Плавность (макс. градиент):  {grad_inside.max():.4f} °C/пиксель")
    
    print("\n📏 ОШИБКИ РЕКОНСТРУКЦИИ:")
    abs_error = np.abs(valid_true - valid_recon)
    print(f"   Средняя (MAE):      {abs_error.mean():.2f} °C")
    print(f"   Максимальная:       {abs_error.max():.2f} °C")
    print(f"   RMSE:               {np.sqrt(np.mean(abs_error**2)):.2f} °C")
    print(f"   Медианная:          {np.median(abs_error):.2f} °C")
    
    print("\n📡 ИЗМЕРЕНИЯ РАДИОМЕТРА:")
    print(f"   Количество антенн:  {len(Tb_data)}")
    print(f"   Tb (мин):           {Tb_noisy.min():.2f} K")
    print(f"   Tb (макс):          {Tb_noisy.max():.2f} K")
    print(f"   Tb (среднее):       {Tb_noisy.mean():.2f} K")
    print(f"   Emissivity (средн.): {emissivity_avg.mean():.3f} ± {emissivity_avg.std():.3f}")
    
    print("\n🎯 ДЕТЕКЦИЯ ОПУХОЛИ:")
    if hasattr(model, 'tumor_center') and model.tumor_center:
        ty, tx = model.tumor_center
        y_coords, x_coords = np.ogrid[:temp_true.shape[0], :temp_true.shape[1]]
        tumor_region = (x_coords - tx)**2 + (y_coords - ty)**2 <= 10**2
        
        if np.sum(tumor_region & breast_mask) > 0:
            tumor_true = temp_true[tumor_region & breast_mask]
            tumor_recon = temp_recon[tumor_region & breast_mask]
            print(f"   Координаты:         Y={ty}, X={tx}")
            print(f"   T в опухоли (истина): {tumor_true.mean():.2f} °C")
            print(f"   T в опухоли (рекон):  {tumor_recon.mean():.2f} °C")
            print(f"   Контраст опухоли:     {tumor_true.mean() - valid_true.mean():.2f} °C")
        else:
            print("   Опухоль за пределами груди")
    else:
        print("   Опухоль не была создана")
    
    if tissue_type_map is not None:
        print("\n🧬 ТЕМПЕРАТУРЫ ПО ТИПАМ ТКАНЕЙ:")
        tissue_names = {1: 'Подкожный жир', 2: 'Железистая', 3: 'Внутрижелез. жир',
                       4: 'Ретромаммарный', 5: 'Соединительная', 6: 'Протоки', 7: 'Дольки'}
        for i, name in tissue_names.items():
            mask = tissue_type_map == i
            if np.sum(mask) > 100:
                t_mean = np.mean(temp_true[mask])
                t_std = np.std(temp_true[mask])
                pct = np.sum(mask) / np.sum(breast_mask) * 100
                print(f"   {name:15s}: {t_mean:5.2f} ± {t_std:.2f} °C ({pct:5.1f}%)")
    
    print("\n" + "="*70)

# =============================================================================
# 🚀 ОСНОВНАЯ ПРОГРАММА
# =============================================================================
if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    print("="*70)
    print("🔬 МОДЕЛЬ РАДИОМЕТРИИ МОЛОЧНОЙ ЖЕЛЕЗЫ (ВЫСОКОЕ РАЗРЕШЕНИЕ)")
    print("="*70)
    
    # 🔥 ВЫБЕРИТЕ РАЗРЕШЕНИЕ:
    RESOLUTION_PRESETS = {
        'low': {'shape': (80, 100), 'tumor_radius': 8, 'resolution_mm': 4},
        'medium': {'shape': (160, 200), 'tumor_radius': 12, 'resolution_mm': 2},
        'high': {'shape': (320, 400), 'tumor_radius': 16, 'resolution_mm': 1},
        'ultra': {'shape': (480, 600), 'tumor_radius': 20, 'resolution_mm': 0.5}
    }
    
    # 🔥 Измените здесь для разного разрешения:
    quality = 'medium'  # 'low', 'medium', 'high', 'ultra'
    
    preset = RESOLUTION_PRESETS[quality]
    print(f"\n📐 Режим: {quality.upper()} ({preset['shape'][0]}×{preset['shape'][1]} пикселей)")
    
    birads = 'B'
    model = BreastRadiometryModelReal(
        freq_ghz=3.0, 
        resolution_mm=preset['resolution_mm'], 
        birads_category=birads
    )
    
    print("\n📌 Генерация анатомического фантома...")
    start_total = time.time()
    
    tumor_position = (int(preset['shape'][0] * 0.56), int(preset['shape'][1] * 0.5))
    
    eps_map, cond_map, temp_true, breast_mask, areola_mask, nipple_mask, body_mask, tissue_type_map = model.create_anatomical_phantom(
        shape=preset['shape'], 
        tumor_radius=preset['tumor_radius'],
        tumor_pos=tumor_position
    )
    
    h, w = temp_true.shape
    scan_y = int(h * 0.45)
    x_pos = np.linspace(int(w * 0.25), int(w * 0.75), 25, dtype=int)
    scan_grid = [(scan_y, x) for x in x_pos]
    
    print(f"📡 Количество антенн: {len(scan_grid)}")
    print(f"📍 Позиция сканера: Y={scan_y}")
    
    print("\n📡 Выполнение прямого сканирования...")
    Tb_data, emissivity_avg = model.forward_scan(eps_map, cond_map, temp_true, breast_mask, scan_grid)
    Tb_noisy = Tb_data + np.random.normal(0, 0.10, size=Tb_data.shape)
    
    print("🔄 Реконструкция температуры...")
    temp_recon = model.reconstruct_simple(Tb_noisy, emissivity_avg, scan_grid, temp_true.shape, breast_mask)
    
    print("\n📊 Генерация графиков...")
    
    # (все функции визуализации из предыдущего кода)
    plot_main_results(temp_true, temp_recon, breast_mask, model.tumor_center, areola_mask, nipple_mask, body_mask)
    plot_tissue_composition(tissue_type_map, breast_mask, areola_mask, nipple_mask, body_mask, birads)
    plot_breast_anatomy(eps_map, breast_mask, areola_mask, nipple_mask, body_mask)
    plot_temperature_gradient(temp_true, breast_mask, model.tumor_center)
    plot_temperature_contours(temp_true, breast_mask, model.tumor_center)
    plot_temperature_difference_map(temp_true, tissue_type_map, breast_mask)
    plot_sensitivity_kernels(model, breast_mask, scan_grid, n_show=5)
    plot_measurement_data(Tb_data, Tb_noisy, emissivity_avg, scan_grid)
    plot_temperature_histogram(temp_true, temp_recon, breast_mask)
    plot_cross_section(temp_true, temp_recon, breast_mask, model.tumor_center)
    plot_emissivity_map(eps_map, breast_mask)
    
    print_full_statistics(temp_true, temp_recon, breast_mask, Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, model, tissue_type_map)
    
    total_elapsed = time.time() - start_total
    print(f"\n⏱️ Общее время выполнения: {total_elapsed:.2f} сек ({total_elapsed/60:.1f} мин)")
    print("\n✅ Все графики сохранены в файлы 01_*.png ... 11_*.png")
    print("="*70)