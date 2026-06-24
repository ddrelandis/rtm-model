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

    def create_anatomical_phantom(self, shape=(100, 160, 160), tumor_radius=10):
        """
        Создает 3D фантом молочной железы.
        shape: (depth, height, width) - z, y, x
        """
        start_time = time.time()
        d, h, w = shape
        print(f"🔍 Генерация 3D фантома: {d}x{h}x{w} вокселей")

        # 1. Создаем сетку координат
        z, y, x = np.ogrid[:d, :h, :w]
        
        # 2. Базовая форма груди (полуэллипсоид)
        # Центр эллипсоида смещаем по Z (глубине) и Y (высоте)
        center_z, center_y, center_x = d * 0.2, h * 0.5, w * 0.5
        
        # Радиусы эллипсоида
        rz, ry, rx = d * 0.8, h * 0.45, w * 0.45
        
        # Уравнение эллипсоида: (z-cz)^2/rz^2 + ... <= 1
        breast_mask = ((z - center_z)**2 / rz**2 + 
                       (y - center_y)**2 / ry**2 + 
                       (x - center_x)**2 / rx**2) <= 1.0
        
        # Обрезаем "лишнее" сзади (тело) и сверху (воздух), если нужно
       # Отсекаем заднюю часть по оси Z (глубине) с помощью среза
        breast_mask[int(d * 0.6):, :, :] = False

        # 3. Слои кожи (Skin)
        skin_thickness = max(2, int(2 * (h/160))) # Масштабируем толщину
        skin_mask = binary_erosion(breast_mask, iterations=skin_thickness) ^ breast_mask
        inner_breast = binary_erosion(breast_mask, iterations=skin_thickness)

        # 4. Железистая ткань и жир (упрощенно для старта)
        # В центре больше железистой ткани, по краям - жир
        dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
        glandular_mask = inner_breast & (dist_from_center < np.mean([ry, rx]) * 0.7)
        fat_mask = inner_breast & ~glandular_mask

        # 5. Опухоль (3D сфера Гаусса)
        tumor_z, tumor_y, tumor_x = int(d*0.3), int(h*0.5), int(w*0.5) # По центру
        dist_to_tumor = np.sqrt((z - tumor_z)**2 + (y - tumor_y)**2 + (x - tumor_x)**2)
        tumor_mask = dist_to_tumor <= tumor_radius
        
        # 6. Заполнение массивов значениями
        eps_map = np.zeros(shape, dtype=np.float32)
        cond_map = np.zeros(shape, dtype=np.float32)
        temp_map = np.zeros(shape, dtype=np.float32)
        
        # Базовые значения (фон/воздух)
        eps_map[:] = 1.0
        temp_map[:] = 20.0
        
        # Жиры
        if np.any(fat_mask):
            eps_map[fat_mask] = np.random.normal(5.0, 0.5, np.sum(fat_mask))
            temp_map[fat_mask] = 35.0
            
        # Железистая ткань
        if np.any(glandular_mask):
            eps_map[glandular_mask] = np.random.normal(45.0, 5.0, np.sum(glandular_mask))
            temp_map[glandular_mask] = 35.5
            
        # Кожа
        if np.any(skin_mask):
            eps_map[skin_mask] = np.random.normal(35.0, 4.0, np.sum(skin_mask))
            temp_map[skin_mask] = 33.8
            
        # Опухоль (переписываем значения внутри маски опухоли)
        if np.any(tumor_mask):
            eps_map[tumor_mask] = np.random.normal(55.0, 7.0, np.sum(tumor_mask))
            # Температура опухоли выше + градиент от центра
            temp_map[tumor_mask] = 38.0 + 1.5 * np.exp(-dist_to_tumor[tumor_mask]**2 / (2 * (tumor_radius*0.5)**2))

        # 7. Гладкость (фильтрация)
        # Важно: sigma должна быть небольшой, чтобы не размыть границы слишком сильно
        sigma = max(1.0, 1.0 * (h/160))
        temp_map = gaussian_filter(temp_map, sigma=sigma)
        eps_map = gaussian_filter(eps_map, sigma=sigma)
        
        # Возвращаем маску груди отдельно, чтобы знать, где "ткань", а где "пустота"
        return eps_map, cond_map, temp_map, breast_mask, tumor_mask

    def compute_sensitivity_kernel_3d(self, mask, ant_pos_3d):
        """
        Вычисляет 3D ядро чувствительности для одной антенны.
        ant_pos_3d: кортеж (z_ant, y_ant, x_ant)
        """
        d, h, w = mask.shape
        z, y, x = np.ogrid[:d, :h, :w]
        
        # Расстояния в вокселях
        dist_xy = np.sqrt((x - ant_pos_3d[2])**2 + (y - ant_pos_3d[1])**2)
        dist_z = np.abs(z - ant_pos_3d[0])
        
        # Параметры гауссианы (в вокселях). 
        # sigma_xy отвечает за lateral resolution, sigma_z - за глубину проникновения
        sigma_xy = 18.0  
        sigma_z = 25.0   
        
        # 3D Гауссиана
        kernel = np.exp(-(dist_xy**2) / (2 * sigma_xy**2) - (dist_z**2) / (2 * sigma_z**2))
        
        # Учитываем только ткань (маску)
        kernel *= mask
        
        # Нормировка ядра (сумма весов = 1)
        sum_k = np.sum(kernel)
        if sum_k > 0:
            kernel /= sum_k
            
        return kernel

    def forward_scan_3d(self, temp_map, mask, scan_positions_3d):
        """
        Прямое сканирование: расчет яркостной температуры (Tb) для каждой антенны.
        """
        temp_kelvin = temp_map + 273.15
        Tb_data = []
        
        print(f"   📡 Сканирование {len(scan_positions_3d)} антеннами...")
        for i, pos in enumerate(scan_positions_3d):
            kernel = self.compute_sensitivity_kernel_3d(mask, pos)
            # Интегрирование температуры по объему с весом ядра
            Tb = np.sum(kernel * temp_kelvin) 
            Tb_data.append(Tb)
            
        return np.array(Tb_data)

    def reconstruct_3d(self, Tb_data, scan_positions_3d, shape, mask):
        """
        3D Реконструкция методом обратного проецирования (Back-projection).
        """
        recon_kelvin = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        
        print(f"   🔄 Реконструкция 3D объема...")
        for i, pos in enumerate(scan_positions_3d):
            kernel = self.compute_sensitivity_kernel_3d(mask, pos)
            recon_kelvin += kernel * Tb_data[i]
            weight_sum += kernel
            
        # Избегаем деления на ноль
        weight_sum[weight_sum == 0] = 1.0
        
        # Переводим обратно в Цельсии
        recon_celsius = (recon_kelvin / weight_sum) - 273.15
        
        # ✅ ИСПРАВЛЕНИЕ: Сглаживаем ДО обнуления воздуха, чтобы NaN не расплывались!
        recon_celsius = gaussian_filter(recon_celsius, sigma=1.5)
        
        # ✅ ТЕПЕРЬ обнуляем воздух
        recon_celsius[~mask] = np.nan 
        
        return recon_celsius

# Тестовый запуск
if __name__ == "__main__":
    model = BreastRadiometryModel3D()
    eps, cond, temp, mask, tumor = model.create_anatomical_phantom(shape=(80, 120, 120))
    print(f"Размер массива температуры: {temp.shape}")
    print(f"Максимальная температура: {temp.max():.2f} C")