import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, distance_transform_edt
import time

class BreastRadiometryModel3D:
    def __init__(self, freq_ghz=3.5, resolution_mm=2, birads_category='B'):
        self.freq = freq_ghz * 1e9
        self.c = 3e8
        self.res = resolution_mm / 1000.0
        self.birads_density = {'A': (0.10, 0.25), 'B': (0.26, 0.50), 'C': (0.51, 0.75), 'D': (0.76, 0.90)}
        
        # Свойства тканей (те же, что и в 2D, но теперь они будут применяться к вокселям)
        self.tissue_props = {
            'fat': {'mean_eps': 5.0, 'std_eps': 0.5, 'mean_cond': 0.10, 'temp_base': 35.0},
            'gland': {'mean_eps': 45.0, 'std_eps': 5.0, 'mean_cond': 2.4, 'temp_base': 35.0},
            'tumor': {'mean_eps': 55.0, 'std_eps': 7.0, 'mean_cond': 4.0, 'temp_base': 38.0},
            'skin': {'mean_eps': 35.0, 'std_eps': 4.0, 'mean_cond': 1.0, 'temp_base': 33.8},
            # ... можно добавить остальные ткани из твоего 2D кода
        }

    
    def create_anatomical_phantom(self, shape=(100, 160, 160), tumor_radius=10, tumor_pos=None, air_buffer_z=25):
        """
        Создает 3D фантом молочной железы.
        shape: (depth, height, width) - z, y, x
        tumor_pos: кортеж (z, y, x) или None для случайной позиции
        air_buffer_z: количество "воздушных" вокселей над грудью (для антенн)
        """
        start_time = time.time()
        d, h, w = shape
        print(f"🔍 Генерация 3D фантома: {d}x{h}x{w} вокселей (буфер: {air_buffer_z})")

        # 1. Создаем сетку координат
        z, y, x = np.ogrid[:d, :h, :w]
        
        # 2. Базовая форма груди (полуэллипсоид)
        # ✅ ИСПРАВЛЕНИЕ: Центр груди смещаем вниз, чтобы над ней был буфер для антенн
        center_z = air_buffer_z + (d - air_buffer_z) * 0.3
        center_y, center_x = h * 0.5, w * 0.5
        
        rz = (d - air_buffer_z) * 0.7
        ry, rx = h * 0.45, w * 0.45
        
        breast_mask = ((z - center_z)**2 / rz**2 + 
                       (y - center_y)**2 / ry**2 + 
                       (x - center_x)**2 / rx**2) <= 1.0
        
        # Отсекаем заднюю часть по оси Z
        breast_mask[int(d * 0.85):, :, :] = False

        # 3. Слои кожи
        skin_thickness = max(2, int(2 * (h/160)))
        skin_mask = binary_erosion(breast_mask, iterations=skin_thickness) ^ breast_mask
        inner_breast = binary_erosion(breast_mask, iterations=skin_thickness)

        # 4. Железистая ткань и жир
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        glandular_mask = inner_breast & (dist_from_center < np.mean([ry, rx]) * 0.7)
        fat_mask = inner_breast & ~glandular_mask

        # 5. Определяем позицию опухоли
        if tumor_pos is not None:
            tumor_z, tumor_y, tumor_x = tumor_pos
            if not (0 <= tumor_z < d and 0 <= tumor_y < h and 0 <= tumor_x < w):
                print(f"   ⚠️ Позиция опухоли ({tumor_z}, {tumor_y}, {tumor_x}) вне сетки! Генерация случайной...")
                tumor_pos = None
            elif not glandular_mask[tumor_z, tumor_y, tumor_x]:
                print(f"   ⚠️ Позиция ({tumor_z}, {tumor_y}, {tumor_x}) вне железистой ткани! Коррекция...")
                gz, gy, gx = np.where(glandular_mask)
                if len(gz) > 0:
                    dists = np.sqrt((gz - tumor_z)**2 + (gy - tumor_y)**2 + (gx - tumor_x)**2)
                    nearest_idx = np.argmin(dists)
                    tumor_z, tumor_y, tumor_x = gz[nearest_idx], gy[nearest_idx], gx[nearest_idx]
                    print(f"   ✅ Скорректированная позиция: Z={tumor_z}, Y={tumor_y}, X={tumor_x}")
                else:
                    print("   ❌ Не удалось скорректировать позицию. Генерация случайной...")
                    tumor_pos = None
        
        if tumor_pos is None:
            valid_z, valid_y, valid_x = np.where(
                glandular_mask & (z > air_buffer_z) & (z < d*0.6)
            )
            if len(valid_z) > 0:
                idx = np.random.randint(0, len(valid_z))
                tumor_z, tumor_y, tumor_x = valid_z[idx], valid_y[idx], valid_x[idx]
                print(f"   🎯 Опухоль создана в случайной позиции: Z={tumor_z}, Y={tumor_y}, X={tumor_x}")
            else:
                print("   ❌ Не удалось найти позицию для опухоли")
                tumor_z, tumor_y, tumor_x = int(center_z), int(center_y), int(center_x)
                print(f"   🎯 Используем центр груди: Z={tumor_z}, Y={tumor_y}, X={tumor_x}")
        else:
            print(f"   🎯 Опухоль создана в заданной позиции: Z={tumor_z}, Y={tumor_y}, X={tumor_x}")

        # 6. Создаем маску опухоли
        dist_to_tumor = np.sqrt((z - tumor_z)**2 + (y - tumor_y)**2 + (x - tumor_x)**2)
        tumor_mask = dist_to_tumor <= tumor_radius
        
        # 7. Заполнение массивов
        eps_map = np.zeros(shape, dtype=np.float32)
        cond_map = np.zeros(shape, dtype=np.float32)
        temp_map = np.zeros(shape, dtype=np.float32)
        
        eps_map[:] = 1.0
        temp_map[:] = 20.0
        
        if np.any(fat_mask):
            eps_map[fat_mask] = np.random.normal(5.0, 0.5, np.sum(fat_mask))
            cond_map[fat_mask] = np.random.normal(0.10, 0.03, np.sum(fat_mask))
            temp_map[fat_mask] = 35.0
            
        if np.any(glandular_mask):
            eps_map[glandular_mask] = np.random.normal(45.0, 5.0, np.sum(glandular_mask))
            cond_map[glandular_mask] = np.random.normal(2.4, 0.4, np.sum(glandular_mask))
            temp_map[glandular_mask] = 35.5
            
        if np.any(skin_mask):
            eps_map[skin_mask] = np.random.normal(35.0, 4.0, np.sum(skin_mask))
            cond_map[skin_mask] = np.random.normal(1.0, 0.2, np.sum(skin_mask))
            temp_map[skin_mask] = 33.8
            
        if np.any(tumor_mask):
            eps_map[tumor_mask] = np.random.normal(55.0, 7.0, np.sum(tumor_mask))
            cond_map[tumor_mask] = np.random.normal(4.0, 0.8, np.sum(tumor_mask))
            temp_map[tumor_mask] = 38.0 + 1.5 * np.exp(-dist_to_tumor[tumor_mask]**2 / (2 * (tumor_radius*0.5)**2))

        # 8. Температурный градиент
        dist_from_surface = distance_transform_edt(~breast_mask).astype(np.float32)
        dist_from_surface[~breast_mask] = 0
        max_dist = dist_from_surface[breast_mask].max()
        if max_dist > 0:
            normalized_depth = dist_from_surface / max_dist
            temp_map += 1.5 * (normalized_depth ** 0.6) * breast_mask
        
        # 9. Воспалительная зона
        inflammation_radius = tumor_radius * 2.5
        temp_map += 0.8 * np.exp(-dist_to_tumor**2 / (2 * inflammation_radius**2)) * breast_mask

        # 10. Гладкость
        sigma = max(1.0, 1.0 * (h/160))
        temp_map = gaussian_filter(temp_map, sigma=sigma)
        eps_map = gaussian_filter(eps_map, sigma=sigma)
        cond_map = gaussian_filter(cond_map, sigma=sigma)
        
        temp_map[~breast_mask] = 20.0
        eps_map[~breast_mask] = 1.0
        cond_map[~breast_mask] = 0.0
        
        temp_map = np.clip(temp_map, 34.0, 39.5)
        
        print(f"   ✅ Время создания фантома: {time.time() - start_time:.2f} сек")
        
        return eps_map, cond_map, temp_map, breast_mask, tumor_mask

    def compute_emissivity_3d(self, eps_map, mask=None):
        """
        Расчет коэффициента излучения (emissivity) для 3D объема.
        Полное соответствие с 2D версией из model.py.
        """
        sqrt_eps = np.sqrt(np.maximum(eps_map, 1.0))
        gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
        emissivity_fresnel = 1.0 - gamma**2
        
        emissivity = 0.88 + 0.11 * (emissivity_fresnel - 0.5) / 0.5
        
        # ✅ КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: clip [0.98, 0.99] как в 2D версии!
        emissivity = np.clip(emissivity, 0.98, 0.99)
        
        np.random.seed(42)
        noise = np.random.normal(0, 0.015, emissivity.shape)
        if mask is not None:
            emissivity = np.clip(emissivity + noise * mask, 0.90, 0.99)
        else:
            emissivity = np.clip(emissivity + noise, 0.90, 0.99)
        
        return emissivity

    def compute_sensitivity_kernel_3d(self, mask, ant_pos_3d):
        """
        Вычисляет 3D ядро чувствительности для одной антенны.
        Антенна смотрит ВНИЗ (в направлении увеличения Z).
        """
        d, h, w = mask.shape
        z, y, x = np.ogrid[:d, :h, :w]
        
        # Расстояния в вокселях
        dist_xy = np.sqrt((x - ant_pos_3d[2])**2 + (y - ant_pos_3d[1])**2)
        
        # ✅ ИСПРАВЛЕНИЕ: Антенна смотрит только ВНИЗ (Z > ant_pos_3d[0])
        # Слои ВЫШЕ антенны (Z < ant_pos_3d[0]) не видны
        z_relative = z - ant_pos_3d[0]  # Положительное = ниже антенны
        
        # Параметры гауссианы
        sigma_xy = 20.0  
        sigma_z = 45.0   
        
        # 3D Гауссиана только для Z >= ant_pos_3d[0]
        # Используем max(0, z_relative) чтобы обнулить область выше антенны
        z_valid = np.maximum(z_relative, 0)
        
        kernel = np.exp(-(dist_xy**2) / (2 * sigma_xy**2) - (z_valid**2) / (2 * sigma_z**2))
        
        # ✅ ДОБАВЛЯЕМ сильный depth_weight (усиливаем глубокие слои)
        # Чем глубже (больше Z), тем сильнее сигнал
        depth_weight = np.clip(1.0 + z_relative / 30.0, 0.0, 3.0)
        kernel *= depth_weight
        
        # ✅ ОБНУЛЯЕМ область выше антенны (Z < ant_pos_3d[0])
        # Используем срез вместо булевой индексации, т.к. z из np.ogrid имеет форму (d,1,1)
        kernel[:int(ant_pos_3d[0]), :, :] = 0.0
        
        # Учитываем только ткань (маску)
        kernel *= mask
        
        # Нормировка ядра (сумма весов = 1)
        sum_k = np.sum(kernel)
        if sum_k > 0:
            kernel /= sum_k
            
        return kernel

    
    def forward_scan_3d(self, temp_map, eps_map, mask, scan_positions_3d):
        """
        Прямое сканирование: расчет яркостной температуры (Tb).
        Формула: Tb = sum(kernel * emissivity * T_kelvin)
        Полное соответствие с 2D версией из model.py.
        """
        temp_kelvin = temp_map + 273.15
        emissivity = self.compute_emissivity_3d(eps_map, mask)
        
        measurements = []
        emissivity_avg = []
        
        print(f"   📡 Сканирование {len(scan_positions_3d)} антеннами...")
        for i, pos in enumerate(scan_positions_3d):
            kernel = self.compute_sensitivity_kernel_3d(mask, pos)
            
            # ✅ Точная формула из 2D: sum(kernel * emissivity * T_kelvin)
            emissivity_avg.append(np.sum(kernel * emissivity))
            measurements.append(np.sum(kernel * emissivity * temp_kelvin))
            
        return np.array(measurements), np.array(emissivity_avg)

    def reconstruct_3d(self, Tb_data, emissivity_avg, scan_positions_3d, shape, mask):
        """
        3D Реконструкция методом обратного проецирования.
        """
        recon_kelvin = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        
        print(f"   🔄 Реконструкция 3D объема...")
        for i, pos in enumerate(scan_positions_3d):
            kernel = self.compute_sensitivity_kernel_3d(mask, pos)
            
            emissivity_corr = emissivity_avg[i] if emissivity_avg[i] > 0.5 else 0.95
            Tb_corrected = Tb_data[i] / emissivity_corr
            
            recon_kelvin += kernel * Tb_corrected
            weight_sum += kernel
            
        weight_sum[weight_sum == 0] = 1.0
        recon_celsius = (recon_kelvin / weight_sum) - 273.15
        
        # ✅ ИСПРАВЛЕНИЕ: Убираем жесткий percentile clip [2, 98]
        # Он обрезал пики опухоли! Оставляем только мягкий clip
        recon_celsius = np.clip(recon_celsius, 32.0, 42.0)
        
        # ✅ УМЕНЬШАЕМ сглаживание (было 1.5, стало 0.5)
        # Это сохранит резкость горячей зоны опухоли
        recon_celsius = gaussian_filter(recon_celsius, sigma=0.5)
        
        # Обнуляем воздух
        recon_celsius[~mask] = np.nan
        
        return recon_celsius
    
# Тестовый запуск
if __name__ == "__main__":
    model = BreastRadiometryModel3D()
    eps, cond, temp, mask, tumor = model.create_anatomical_phantom(shape=(80, 120, 120))
    print(f"Размер массива температуры: {temp.shape}")
    print(f"Максимальная температура: {temp.max():.2f} C")