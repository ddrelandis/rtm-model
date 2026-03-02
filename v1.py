import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class BreastRadiometryModel:
    def __init__(self, freq_ghz=3.0, resolution_mm=5):
        self.freq = freq_ghz * 1e9
        self.c = 3e8
        self.lambda0 = self.c / self.freq
        self.res = resolution_mm / 1000.0  # в метрах
        
        # Диэлектрические свойства (упрощенные данные Gabriel et al.)
        self.tissue_props = {
            'fat': {'eps': 12.0, 'cond': 0.8, 'temp': 35.0},
            'gland': {'eps': 45.0, 'cond': 2.5, 'temp': 36.5},
            'tumor': {'eps': 55.0, 'cond': 4.0, 'temp': 37.0}  # Гипертермия
        }

    def create_anthropomorphic_phantom(self, shape=(50, 50), tumor_pos=(15, 25), tumor_radius=3):
        """
        Создание синтетического фантома молочной железы
        shape: размер сетки в ячейках
        """
        # Базовая ткань (жир)
        eps_map = np.ones(shape) * self.tissue_props['fat']['eps']
        cond_map = np.ones(shape) * self.tissue_props['fat']['cond']
        temp_map = np.ones(shape) * self.tissue_props['fat']['temp']
        
        # Добавление фиброгландулярных включений (шум)
        noise = np.random.rand(*shape) > 0.1
        eps_map[noise] = self.tissue_props['gland']['eps']
        cond_map[noise] = self.tissue_props['gland']['cond']
        temp_map[noise] = self.tissue_props['gland']['temp']
        
        # Добавление опухоли
        y, x = np.ogrid[:shape[0], :shape[1]]
        mask = (x - tumor_pos[0])**2 + (y - tumor_pos[1])**2 <= tumor_radius**2
        eps_map[mask] = self.tissue_props['tumor']['eps']
        cond_map[mask] = self.tissue_props['tumor']['cond']
        temp_map[mask] = self.tissue_props['tumor']['temp']
        
        return eps_map, cond_map, temp_map

    def compute_sensitivity_kernel(self, eps_map, ant_pos):
        """
        Расчет функции чувствительности антенны в точке ant_pos.
        Упрощение: Гауссово пятно с затуханием, зависящим от проводимости.
        В реальности здесь решается уравнение Максвелла (FDTD).
        """
        h, w = eps_map.shape
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - ant_pos[1])**2 + (y - ant_pos[0])**2) * self.res
        
        # Глубина проникновения (скин-слой) зависит от частоты и проводимости
        # delta = 1 / sqrt(pi * f * mu * sigma)
        sigma_avg = np.mean(self.tissue_props['gland']['cond']) 
        delta = 1.0 / np.sqrt(np.pi * self.freq * 4*np.pi*1e-7 * sigma_avg)
        
        # Чувствительность падает с глубиной и расстоянием от антенны
        sensitivity = np.exp(-dist / (self.lambda0 / 4)) * np.exp(-dist / delta)
        return sensitivity / np.sum(sensitivity)

    def forward_scan(self, eps_map, cond_map, temp_map, scan_positions):
        """
        Моделирование сканирования антенной решеткой
        scan_positions: список координат (y, x) антенн
        """
        measurements = []
        for pos in scan_positions:
            kernel = self.compute_sensitivity_kernel(eps_map, pos)
            # Интеграл свертки: Tb = ∫ K(r) * T(r) * emissivity(r) dr
            # Коэффициент излучения упрощенно считаем зависящим от импеданса
            emissivity = 1.0 - np.abs((np.sqrt(eps_map) - 1)/(np.sqrt(eps_map) + 1))**2
            
            Tb = np.sum(kernel * temp_map * emissivity)
            measurements.append(Tb)
            
        return np.array(measurements)

    def reconstruct_tikhonov_2d(self, measurements, scan_positions, shape, alpha=1e-2):
        """
        Упрощенная реконструкция 2D поля (линеаризованная)
        Для реальной задачи лучше использовать итеративные методы (DBIM)
        """
        N_meas = len(measurements)
        N_pixels = shape[0] * shape[1]
        
        # Построение матрицы чувствительности (очень затратно для больших сеток)
        # Для демо используем упрощенный бэк-проекции с регуляризацией
        recon_field = np.zeros(shape)
        
        for i, pos in enumerate(scan_positions):
            # Создаем "луч" от позиции измерения
            kernel = self.compute_sensitivity_kernel(np.ones(shape), pos)
            # Обратное проецирование взвешенное
            recon_field += kernel * measurements[i]
            
        # Сглаживание регуляризацией (аналог Тихонова в пространстве изображений)
        recon_field = gaussian_filter(recon_field, sigma=2.0)
        
        # Нормализация к диапазону температур
        min_t, max_t = np.min(recon_field), np.max(recon_field)
        if max_t > min_t:
            recon_field = 35.0 + (recon_field - min_t) / (max_t - min_t) * 5.0
            
        return recon_field

# --- Пример использования ---
if __name__ == "__main__":
    model = BreastRadiometryModel(freq_ghz=8.0)
    
    # 1. Создание фантома с опухолью
    eps, cond, temp_true = model.create_anthropomorphic_phantom(tumor_pos=(30, 30), tumor_radius=8)
    
    # 2. Сетка сканирования (например, 10x10 антенн над поверхностью)
    h, w = temp_true.shape
    n = 25
    y_pos = np.linspace(5, h-5, n, dtype=int)
    x_pos = np.linspace(5, w-5, n, dtype=int)
    scan_grid = [(y, x) for y in y_pos for x in x_pos]
    
    # 3. Прямая задача (с шумом измерений)
    Tb_data = model.forward_scan(eps, cond, temp_true, scan_grid)
    Tb_noisy = Tb_data + np.random.normal(0, 0.2, size=Tb_data.shape) # Шум радиометра 0.2 К
    
    # 4. Обратная задача
    temp_recon = model.reconstruct_tikhonov_2d(Tb_noisy, scan_grid, temp_true.shape)
    
    # 5. Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    im1 = axes[0].imshow(temp_true, cmap='jet', vmin=34, vmax=40)
    axes[0].set_title('Истинное распределение T')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(temp_recon, cmap='jet', vmin=34, vmax=40)
    axes[1].set_title('Реконструированное T')
    plt.colorbar(im2, ax=axes[1])
    
    diff = np.abs(temp_true - temp_recon)
    im3 = axes[2].imshow(diff, cmap='Reds', vmin=0, vmax=2)
    axes[2].set_title('Ошибка реконструкции')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()