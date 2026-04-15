import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, distance_transform_edt
import time

class BreastRadiometryModelReal:
    def __init__(self, freq_ghz=3.0, resolution_mm=2, birads_category='B', temp_vmin=None, temp_vmax=None):
        """
        Инициализация модели радиометрии молочной железы.
        """
        self.freq = freq_ghz * 1e9
        self.omega = 2 * np.pi * self.freq
        self.c = 3e8
        self.lambda0 = self.c / self.freq
        self.res = resolution_mm / 1000.0
        self.eps0 = 8.854e-12
        self.tumor_center = None
        self.birads_category = birads_category
        
        # Плотность тканей по BI-RADS
        self.birads_density = {
            'A': (0.10, 0.25),
            'B': (0.26, 0.50),
            'C': (0.51, 0.75),
            'D': (0.76, 0.90)
        }
        
        # Температурные границы для визуализации
        self.temp_vmin = temp_vmin if temp_vmin is not None else 33.0
        self.temp_vmax = temp_vmax if temp_vmax is not None else 40.0
        
        # Свойства тканей (диэлектрические и тепловые)
        self.tissue_props = {
            'fat': {'mean_eps': 5.0, 'std_eps': 0.5, 'mean_cond': 0.10, 'std_cond': 0.03, 'temp_base': 35.0},
            'fat_subcutaneous': {'mean_eps': 4.5, 'std_eps': 0.4, 'mean_cond': 0.08, 'std_cond': 0.02, 'temp_base': 34.8},
            'fat_retromammary': {'mean_eps': 5.0, 'std_eps': 0.5, 'mean_cond': 0.10, 'std_cond': 0.03, 'temp_base': 35.0},
            'gland': {'mean_eps': 45.0, 'std_eps': 5.0, 'mean_cond': 2.4, 'std_cond': 0.4, 'temp_base': 35.5},
            'gland_ducts': {'mean_eps': 48.0, 'std_eps': 5.0, 'mean_cond': 2.8, 'std_cond': 0.5, 'temp_base': 36.0},
            'connective': {'mean_eps': 30.0, 'std_eps': 4.0, 'mean_cond': 1.3, 'std_cond': 0.3, 'temp_base': 35.2},
            'tumor': {'mean_eps': 55.0, 'std_eps': 7.0, 'mean_cond': 4.0, 'std_cond': 0.8, 'temp_base': 38.0},
            'nipple': {'mean_eps': 45.0, 'std_eps': 5.0, 'mean_cond': 2.6, 'std_cond': 0.5, 'temp_base': 35.8},
            'body': {'mean_eps': 50.0, 'std_eps': 5.0, 'mean_cond': 2.0, 'std_cond': 0.3, 'temp_base': 37.0},
            'skin': {'mean_eps': 35.0, 'std_eps': 4.0, 'mean_cond': 1.0, 'std_cond': 0.2, 'temp_base': 33.8}
        }

    def get_tissue_values(self, tissue_type, shape):
        key = tissue_type if isinstance(tissue_type, str) else {
            1: 'fat_subcutaneous', 2: 'gland', 3: 'fat', 4: 'fat_retromammary', 
            5: 'connective', 6: 'gland_ducts', 7: 'gland', 8: 'nipple', 9: 'gland', 10: 'skin', 11: 'body'
        }.get(tissue_type, tissue_type)
        
        props = self.tissue_props[key]
        eps_map = np.clip(np.random.normal(props['mean_eps'], props.get('std_eps', 0), shape), 1.0, None)
        cond_map = np.clip(np.random.normal(props['mean_cond'], props.get('std_cond', 0), shape), 0.01, None)
        # 🔥 Добавлен temp_offset к базовой температуре
        temp_map = np.ones(shape) * (props['temp_base'] + props.get('temp_offset', 0))
        return eps_map, cond_map, temp_map

    def create_anatomical_phantom(self, shape=(160, 200), tumor_radius=12, tumor_pos=None):
        """
        Создание анатомического фантома молочной железы.
        """
        start_time = time.time()
        h, w = shape
        y, x = np.ogrid[:h, :w]
        center_x = w / 2.0
        scale_factor = h / 80.0

        print(f"🔍 Разрешение: {w}×{h} пикселей (масштаб: {scale_factor:.2f}×)")
        
        # === 1. Форма груди ===
        top_y = int(h * 0.12)
        breast_width_top, breast_width_mid, breast_width_base = w * 0.06, w * 0.35, w * 0.55
        breast_mask = np.zeros(shape, dtype=bool)

        for yi in range(top_y, h):
            normalized_y = (yi - top_y) / (h - top_y)
            if normalized_y < 0.25:
                width_factor = breast_width_top + (breast_width_mid - breast_width_top) * (normalized_y / 0.25) ** 0.5
            elif normalized_y < 0.6:
                width_factor = breast_width_mid + (breast_width_base - breast_width_mid) * ((normalized_y - 0.25) / 0.35)
            else:
                width_factor = breast_width_base
            x_left, x_right = max(0, int(center_x - width_factor)), min(w, int(center_x + width_factor))
            breast_mask[yi, x_left:x_right] = True

        breast_mask = binary_dilation(breast_mask, iterations=max(1, int(2 * scale_factor)))
        breast_mask = gaussian_filter(breast_mask.astype(float), sigma=0.8 * scale_factor) > 0.5

        # === 2. Анатомические структуры ===
        nipple_center_y, nipple_center_x = int(h * 0.15), int(w / 2.0)
        areola_radius, nipple_radius = int(w * 0.10), int(w * 0.04)
        areola_mask = ((x - nipple_center_x)**2 + (y - nipple_center_y)**2 <= areola_radius**2) & breast_mask
        nipple_mask = ((x - nipple_center_x)**2 + (y - nipple_center_y)**2 <= nipple_radius**2) & areola_mask

        skin_thickness = max(2, int(2 * scale_factor))
        skin_mask = (binary_erosion(breast_mask, iterations=skin_thickness) ^ breast_mask) & breast_mask
        subcut_thickness = max(4, int(h * 0.06))
        subcutaneous_mask = (binary_erosion(binary_erosion(breast_mask, iterations=skin_thickness), iterations=subcut_thickness) ^ 
                             binary_erosion(breast_mask, iterations=skin_thickness)) & breast_mask & ~skin_mask
        retromammary_mask = (y >= int(h * 0.65)) & breast_mask & ~skin_mask & ~subcutaneous_mask
        glandular_mask = breast_mask & ~skin_mask & ~subcutaneous_mask & ~retromammary_mask & ~areola_mask
        body_mask = (y >= int(h * 0.75)) & breast_mask

        # === 3. Неоднородность железистой ткани ===
        density_range = self.birads_density[self.birads_category]
        target_gland_fraction = np.random.uniform(density_range[0], density_range[1])
        print(f"📊 BI-RADS категория: {self.birads_category}")
        print(f"   Целевая доля железистой ткани: {target_gland_fraction*100:.1f}%")

        n_lobes = np.random.randint(15, 21)
        lobe_mask = np.zeros(shape, dtype=bool)
        gland_center_y, gland_center_x = int(h * 0.45), int(w / 2.0)
        for i in range(n_lobes):
            angle = (2 * np.pi * i) / n_lobes
            lobe_width = np.random.uniform(0.15, 0.25) * np.pi / n_lobes
            dy, dx = y - gland_center_y, x - gland_center_x
            angle_map = np.arctan2(dy, dx)
            angle_diff = np.minimum(np.abs(angle_map - angle), 2*np.pi - np.abs(angle_map - angle))
            sector_mask = (angle_diff < lobe_width) & glandular_mask & (np.sqrt(dx**2 + dy**2) < w * 0.4)
            lobe_mask |= sector_mask

        n_lobules = int(np.sum(lobe_mask) * 0.003)
        lobule_mask = np.zeros(shape, dtype=bool)
        lobule_indices = np.where(lobe_mask)
        for _ in range(n_lobules):
            if len(lobule_indices[0]) == 0: break
            idx = np.random.randint(0, len(lobule_indices[0]))
            cy, cx = lobule_indices[0][idx], lobule_indices[1][idx]
            lobule_size = max(4, int(np.random.randint(4, 10) * scale_factor))
            yy, xx = np.ogrid[:h, :w]
            lobule_mask |= ((xx - cx)**2 + (yy - cy)**2 <= lobule_size**2) & lobe_mask

        n_ducts = min(n_lobes, 12)
        duct_mask = np.zeros(shape, dtype=bool)
        for i in range(n_ducts):
            angle = (2 * np.pi * i) / n_ducts + np.random.uniform(-0.2, 0.2)
            t = np.linspace(0, 1, 100)
            for ti in t:
                px, py = int(nipple_center_x + ti * (w * 0.35) * np.cos(angle)), int(nipple_center_y + ti * (h * 0.4) * np.sin(angle))
                if 0 <= px < w and 0 <= py < h:
                    duct_radius = max(3, int(3 * scale_factor))
                    duct_mask |= ((x - px)**2 + (y - py)**2 <= duct_radius**2) & glandular_mask

        connective_mask = np.zeros(shape, dtype=bool)
        n_fibers = max(60, int(60 * scale_factor))
        for _ in range(n_fibers):
            fy = np.random.randint(top_y, h)
            fx = np.random.randint(int(center_x - breast_width_base), int(center_x + breast_width_base))
            fiber_length = max(10, int(np.random.randint(10, 25) * scale_factor))
            fiber_angle = np.random.uniform(0, 2*np.pi)
            for l in range(fiber_length):
                px, py = int(fx + l * np.cos(fiber_angle)), int(fy + l * np.sin(fiber_angle))
                if 0 <= px < w and 0 <= py < h:
                    fiber_radius = max(2, int(2 * scale_factor))
                    connective_mask |= ((x - px)**2 + (y - py)**2 <= fiber_radius**2) & glandular_mask

        # === 4. Распределение железистой ткани ===
        available_gland_area = np.sum(glandular_mask)
        target_gland_area = int(available_gland_area * target_gland_fraction)
        gland_priority = np.zeros(shape)
        gland_priority[lobule_mask] = 3
        gland_priority[duct_mask] = 2
        gland_priority[lobe_mask] = 1

        gland_indices = np.where(glandular_mask & (gland_priority > 0))
        final_gland_mask = np.zeros(shape, dtype=bool)
        intragland_fat_mask = glandular_mask.copy()  # Инициализация по умолчанию

        if len(gland_indices[0]) > 0:
            priorities = gland_priority[gland_indices]
            sorted_idx = np.argsort(-priorities)
            gland_filled = 0
            for idx in sorted_idx:
                if gland_filled >= target_gland_area: break
                final_gland_mask[gland_indices[0][idx], gland_indices[1][idx]] = True
                gland_filled += 1
            intragland_fat_mask = glandular_mask & ~final_gland_mask

        # === 5. Заполнение карт свойств ===
        eps_map = np.zeros(shape)
        cond_map = np.zeros(shape)
        temp_map = np.zeros(shape)
        tissue_type_map = np.zeros(shape, dtype=int)

        def fill_mask(mask, t_type, t_id):
            if np.any(mask):
                e, c, t = self.get_tissue_values(t_type, np.sum(mask))
                eps_map[mask] = e
                cond_map[mask] = c
                temp_map[mask] = t
                tissue_type_map[mask] = t_id

        fill_mask(subcutaneous_mask, 'fat_subcutaneous', 1)
        fill_mask(final_gland_mask, 'gland', 2)
        fill_mask(intragland_fat_mask, 'fat', 3)
        fill_mask(retromammary_mask, 'fat_retromammary', 4)
        fill_mask(connective_mask, 'connective', 5)
        fill_mask(duct_mask, 'gland_ducts', 6)
        if np.any(lobule_mask & final_gland_mask):
            temp_map[lobule_mask & final_gland_mask] += 0.9
            tissue_type_map[lobule_mask & final_gland_mask] = 7

        eps_map[~breast_mask], cond_map[~breast_mask], temp_map[~breast_mask] = 1.0, 0.0, 20.0
        if np.any(areola_mask): fill_mask(areola_mask, 'gland', 9)
        if np.any(nipple_mask): fill_mask(nipple_mask, 'nipple', 8)
        if np.any(skin_mask): fill_mask(skin_mask, 'skin', 10)
        if np.any(body_mask): fill_mask(body_mask, 'body', 11)

        # === 6. Температурный градиент ===
        dist_from_surface = distance_transform_edt(~breast_mask).astype(float)
        dist_from_surface[~breast_mask] = 0
        max_dist = dist_from_surface[breast_mask].max()
        normalized_depth = dist_from_surface / max_dist if max_dist > 0 else np.zeros_like(dist_from_surface)
        temp_map += 2.0 * (normalized_depth ** 0.6) * breast_mask
        temp_map += np.random.normal(0, 0.08, shape) * breast_mask
        temp_map = gaussian_filter(temp_map, sigma=max(0.8, 0.8 * scale_factor))
        temp_map = np.clip(temp_map, 34.0, 39.5)
        temp_map[~breast_mask] = 20.0

        # === 7. Опухоль ===
        self.tumor_center = None
        tumor_ty, tumor_tx = None, None
        if tumor_pos is not None and 0 <= tumor_pos[0] < h and 0 <= tumor_pos[1] < w:
            if final_gland_mask[tumor_pos[0], tumor_pos[1]]:
                tumor_ty, tumor_tx = tumor_pos
            else:
                print(f"⚠️ Позиция ({tumor_pos[0]}, {tumor_pos[1]}) вне железистой ткани! Коррекция...")
                y_coords, x_coords = np.where(final_gland_mask)
                if len(y_coords) > 0:
                    dists = np.sqrt((y_coords - tumor_pos[0])**2 + (x_coords - tumor_pos[1])**2)
                    nearest_idx = np.argmin(dists)
                    tumor_ty, tumor_tx = y_coords[nearest_idx], x_coords[nearest_idx]
                    print(f"✅ Скорректированная позиция: Y={tumor_ty}, X={tumor_tx}")
                else:
                    print("⚠️ Не удалось скорректировать позицию")
                    tumor_pos = None

        if tumor_ty is None:
            valid_y, valid_x = np.where(final_gland_mask & (y > h*0.30) & (y < h*0.65))
            if len(valid_y) > 0:
                idx = np.random.randint(0, len(valid_y))
                tumor_ty, tumor_tx = valid_y[idx], valid_x[idx]
                print(f"🎲 Опухоль создана в случайной позиции: Y={tumor_ty}, X={tumor_tx}")
            else:
                print("⚠️ Не удалось найти позицию для опухоли")
                tumor_ty, tumor_tx = None, None

        if tumor_ty is not None:
            print(f"✅ Опухоль создана: Y={tumor_ty}, X={tumor_tx}")
            tumor_y, tumor_x = np.ogrid[:h, :w]
            dist_from_tumor = np.sqrt((tumor_x - tumor_tx)**2 + (tumor_y - tumor_ty)**2)
            tumor_sigma = tumor_radius * 1.5
            
            # Добавление гипертермии и изменения диэлектрических свойств
            temp_map += 2.5 * np.exp(-dist_from_tumor**2 / (2 * tumor_sigma**2)) * breast_mask
            temp_map = np.clip(temp_map, 34.0, 39.5)
            temp_map = gaussian_filter(temp_map, sigma=max(1.0, 1.0 * scale_factor))
            temp_map[~breast_mask] = 20.0
            
            eps_map += 15.0 * np.exp(-dist_from_tumor**2 / (2 * tumor_sigma**2)) * breast_mask
            eps_map = gaussian_filter(eps_map, sigma=max(1.5, 1.5 * scale_factor))
            eps_map[~breast_mask] = 1.0
            
            self.tumor_center = (tumor_ty, tumor_tx)

        print(f"⏱️ Время создания фантома: {time.time() - start_time:.2f} сек")
        return eps_map, cond_map, temp_map, breast_mask, areola_mask, nipple_mask, body_mask, tissue_type_map

    def compute_sensitivity_kernel(self, mask, ant_pos):
        """
        Вычисление ядра чувствительности антенны на основе уравнений Максвелла.
        """
        h, w = mask.shape
        y, x = np.ogrid[:h, :w]
        dx_m = (x - ant_pos[1]) * self.res
        dy_m = (y - ant_pos[0]) * self.res
        r_m = np.sqrt(dx_m**2 + dy_m**2)
        
        # Средние значения для расчёта затухания
        eps_avg = 12.0
        sigma_avg = 1.5
        alpha = self.omega / self.c * np.sqrt(eps_avg / 2 * (np.sqrt(1 + (sigma_avg / (self.omega * self.eps0 * eps_avg))**2) - 1))
        
        # Квазистатическое поле диполя
        r_safe = np.maximum(r_m, self.res * 0.5)
        E_sq = np.exp(-2 * alpha * r_safe) / (r_safe**2 + 1e-6)
        
        # Функция чувствительности
        K = sigma_avg * E_sq * mask
        sum_K = np.sum(K)
        return K / sum_K if sum_K > 0 else K

    def compute_emissivity(self, eps_map, mask=None):
        """
        Расчёт коэффициента излучения по формуле Френеля.
        """
        sqrt_eps = np.sqrt(np.maximum(eps_map, 1.0))
        gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
        emissivity_fresnel = 1.0 - gamma**2
        emissivity = 0.85 + 0.13 * emissivity_fresnel
        noise = np.random.normal(0, 0.012, emissivity.shape)
        if mask is not None:
            emissivity += noise * mask
        return np.clip(emissivity, 0.90, 0.99)

    def forward_scan(self, eps_map, cond_map, temp_map, mask, scan_positions):
        """
        Прямая задача: расчёт яркостной температуры.
        """
        measurements, emissivity_avg = [], []
        temp_kelvin = temp_map + 273.15
        emissivity_map = self.compute_emissivity(eps_map, mask)
        
        for pos in scan_positions:
            K = self.compute_sensitivity_kernel(mask, pos)
            Tb = np.sum(K * emissivity_map * temp_kelvin)
            measurements.append(Tb)
            emissivity_avg.append(np.sum(K * emissivity_map) / (np.sum(K) + 1e-10))
            
        return np.array(measurements), np.array(emissivity_avg)

    def reconstruct_simple(self, measurements, emissivity_avg, scan_positions, shape, mask):
        """
        Обратная задача: реконструкция температурного поля.
        """
        recon_kelvin = np.zeros(shape)
        weight_sum = np.zeros(shape)
        
        # Простое накопление взвешенных измерений
        for i, pos in enumerate(scan_positions):
            kernel = self.compute_sensitivity_kernel(mask, pos)
            e_corr = emissivity_avg[i] if emissivity_avg[i] > 0.5 else 0.95
            Tb_corr = measurements[i] / e_corr
            recon_kelvin += kernel * Tb_corr
            weight_sum += kernel
        
        # 🔥 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Защита от деления на ноль
        with np.errstate(divide='ignore', invalid='ignore'):
            recon_kelvin = np.where(weight_sum > 1e-12, recon_kelvin / weight_sum, np.nan)
        
        recon_celsius = recon_kelvin - 273.15
        
        # Сглаживание только внутри маски
        recon_celsius_smooth = gaussian_filter(
            np.where(mask, recon_celsius, 0), 
            sigma=1.0
        )
        recon_celsius = np.where(mask, recon_celsius_smooth, np.nan)
        
        # Нормализация только по валидным данным
        valid_data = recon_celsius[mask]
        if len(valid_data) > 0 and np.any(np.isfinite(valid_data)):
            p1, p99 = np.percentile(valid_data[np.isfinite(valid_data)], [1, 99])
            recon_celsius = np.clip(recon_celsius, p1 - 0.5, p99 + 0.5)
        
        return recon_celsius