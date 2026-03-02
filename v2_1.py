import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_erosion

class BreastRadiometryModelReal:
    def __init__(self, freq_ghz=3.5, resolution_mm=5):
        self.freq = freq_ghz * 1e9
        self.c = 3e8
        self.lambda0 = self.c / self.freq
        self.res = resolution_mm / 1000.0
        
        self.tissue_props = {
            'fat': {
                'mean_eps': 10.5, 'std_eps': 1.5, 
                'mean_cond': 0.15, 'std_cond': 0.05, 
                'temp': 35.0
            },
            'gland': {
                'mean_eps': 48.0, 'std_eps': 6.0, 
                'mean_cond': 2.6, 'std_cond': 0.4, 
                'temp': 36.5
            },
            'tumor': {
                'mean_eps': 58.0, 'std_eps': 8.0, 
                'mean_cond': 4.2, 'std_cond': 0.8, 
                'temp': 39.5
            }
        }

    def get_tissue_values(self, tissue_type, shape):
        props = self.tissue_props[tissue_type]
        eps_map = np.clip(np.random.normal(props['mean_eps'], props['std_eps'], shape), 1.0, None)
        cond_map = np.clip(np.random.normal(props['mean_cond'], props['std_cond'], shape), 0.01, None)
        temp_map = np.ones(shape) * props['temp']
        return eps_map, cond_map, temp_map

    def create_anatomical_phantom(self, shape=(60, 80), tumor_radius=8):
        h, w = shape
        eps_map = np.zeros(shape)
        cond_map = np.zeros(shape)
        temp_map = np.zeros(shape)
        
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h * 0.8, w / 2.0
        radius_y, radius_x = h * 0.7, w * 0.45
        
        ellipse_mask = ((x - center_x)**2 / radius_x**2 + (y - center_y)**2 / radius_y**2 <= 1) & (y > h * 0.3)
        
        eps_map[ellipse_mask], cond_map[ellipse_mask], temp_map[ellipse_mask] = self.get_tissue_values('fat', np.sum(ellipse_mask))
        eps_map[~ellipse_mask] = 1.0
        cond_map[~ellipse_mask] = 0.0
        temp_map[~ellipse_mask] = 20.0

        n_clusters = 60
        gland_indices = np.where(ellipse_mask)
        for _ in range(n_clusters):
            idx = np.random.randint(0, len(gland_indices[0]))
            cy, cx = gland_indices[0][idx], gland_indices[1][idx]
            cluster_size = np.random.randint(5, 15)
            yy, xx = np.ogrid[:h, :w]
            cluster_mask = (xx - cx)**2 + (yy - cy)**2 <= cluster_size**2
            region_mask = cluster_mask & ellipse_mask
            if np.any(region_mask):
                e, c, t = self.get_tissue_values('gland', np.sum(region_mask))
                eps_map[region_mask] = e
                cond_map[region_mask] = c
                temp_map[region_mask] = t

        valid_y = np.where((ellipse_mask) & (y > h*0.4) & (y < h*0.9))[0]
        valid_x = np.where((ellipse_mask) & (x > w*0.3) & (x < w*0.7))[0]
        
        if len(valid_y) > 0 and len(valid_x) > 0:
            ty = np.random.choice(valid_y)
            tx = np.random.choice(valid_x)
            tumor_mask = (x - tx)**2 + (y - ty)**2 <= tumor_radius**2
            tumor_mask = binary_erosion(tumor_mask, iterations=1) 
            tumor_mask = tumor_mask & ellipse_mask
            
            if np.any(tumor_mask):
                eps_map[tumor_mask], cond_map[tumor_mask], temp_map[tumor_mask] = self.get_tissue_values('tumor', np.sum(tumor_mask))
                self.tumor_center = (ty, tx)  # Сохраняем координаты опухоли
            else:
                self.tumor_center = None
        else:
            self.tumor_center = None

        return eps_map, cond_map, temp_map, ellipse_mask

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
        recon_field = gaussian_filter(recon_field, sigma=3.0)
        
        valid_data = recon_field[mask]
        if len(valid_data) > 0:
            min_t, max_t = np.percentile(valid_data, [5, 95])
            if max_t > min_t:
                recon_field = 35.0 + (recon_field - min_t) / (max_t - min_t) * 4.0
        
        recon_field[~mask] = np.nan
        return recon_field


# =============================================================================
# 📊 ФУНКЦИИ ВИЗУАЛИЗАЦИИ
# =============================================================================

def plot_main_results(temp_true, temp_recon, breast_mask, tumor_center=None):
    """Основные результаты: истина, реконструкция, ошибка"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    temp_display = temp_true.copy()
    temp_display[~breast_mask] = np.nan
    im1 = axes[0].imshow(temp_display, cmap='jet', vmin=34, vmax=40)
    axes[0].set_title('Истинное распределение T', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Пиксели (X)')
    axes[0].set_ylabel('Пиксели (Y)')
    if tumor_center:
        axes[0].plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, markeredgewidth=2, label='Центр опухоли')
        axes[0].legend(loc='upper right')
    plt.colorbar(im1, ax=axes[0], label='Температура (°C)')
    
    im2 = axes[1].imshow(temp_recon, cmap='jet', vmin=34, vmax=40)
    axes[1].set_title('Реконструированное T', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Пиксели (X)')
    if tumor_center:
        axes[1].plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, markeredgewidth=2)
    plt.colorbar(im2, ax=axes[1], label='Температура (°C)')
    
    diff = np.abs(temp_true - temp_recon)
    diff[~breast_mask] = np.nan
    im3 = axes[2].imshow(diff, cmap='magma', vmin=0, vmax=2)
    axes[2].set_title('Абсолютная ошибка', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Пиксели (X)')
    plt.colorbar(im3, ax=axes[2], label='Ошибка (°C)')
    
    plt.tight_layout()
    plt.savefig('01_main_results.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_sensitivity_kernels(model, breast_mask, scan_positions, n_show=5):
    """Показывает функции чувствительности для нескольких антенн"""
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
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax, label='Норм. чувствительность')
    
    plt.suptitle('Функции чувствительности антенн (Sensitivity Kernels)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('02_sensitivity_kernels.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_measurement_data(Tb_data, Tb_noisy, emissivity_avg, scan_positions):
    """График измеренных данных по позициям антенн"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    x_antenna = [pos[1] for pos in scan_positions]
    
    # График Tb
    axes[0].plot(x_antenna, Tb_data, 'bo-', linewidth=2, markersize=8, label='Без шума')
    axes[0].plot(x_antenna, Tb_noisy, 'rs--', linewidth=2, markersize=8, label='С шумом')
    axes[0].set_xlabel('Позиция антенны (X)', fontsize=11)
    axes[0].set_ylabel('Яркостная температура Tb (K)', fontsize=11)
    axes[0].set_title('Измерения яркостной температуры', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # График emissivity
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
    """Гистограммы распределения температур"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    true_vals = temp_true[breast_mask]
    recon_vals = temp_recon[breast_mask]
    
    # Гистограмма истины
    axes[0].hist(true_vals, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(true_vals.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {true_vals.mean():.2f}°C')
    axes[0].set_xlabel('Температура (°C)', fontsize=11)
    axes[0].set_ylabel('Количество пикселей', fontsize=11)
    axes[0].set_title('Распределение истинных температур', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Гистограмма реконструкции
    axes[1].hist(recon_vals, bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(recon_vals.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {recon_vals.mean():.2f}°C')
    axes[1].set_xlabel('Температура (°C)', fontsize=11)
    axes[1].set_ylabel('Количество пикселей', fontsize=11)
    axes[1].set_title('Распределение реконструированных температур', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_temperature_histogram.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_cross_section(temp_true, temp_recon, breast_mask, tumor_center=None):
    """Сравнение профилей температуры через центр опухоли"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    h, w = temp_true.shape
    
    if tumor_center:
        # Горизонтальный срез через центр опухоли
        y_slice = int(tumor_center[0])
        x_range = np.arange(w)
        true_slice = temp_true[y_slice, :]
        recon_slice = temp_recon[y_slice, :]
        mask_slice = breast_mask[y_slice, :]
        
        axes[0].plot(x_range[mask_slice], true_slice[mask_slice], 'b-', linewidth=2, label='Истинная T')
        axes[0].plot(x_range[mask_slice], recon_slice[mask_slice], 'r--', linewidth=2, label='Реконструированная T')
        axes[0].axvline(tumor_center[1], color='green', linestyle=':', linewidth=2, label='Центр опухоли')
        axes[0].set_xlabel('Позиция X (пиксели)', fontsize=11)
        axes[0].set_ylabel('Температура (°C)', fontsize=11)
        axes[0].set_title(f'Горизонтальный срез через Y={y_slice}', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Вертикальный срез
        x_slice = int(tumor_center[1])
        y_range = np.arange(h)
        true_slice_v = temp_true[:, x_slice]
        recon_slice_v = temp_recon[:, x_slice]
        mask_slice_v = breast_mask[:, x_slice]
        
        axes[1].plot(y_range[mask_slice_v], true_slice_v[mask_slice_v], 'b-', linewidth=2, label='Истинная T')
        axes[1].plot(y_range[mask_slice_v], recon_slice_v[mask_slice_v], 'r--', linewidth=2, label='Реконструированная T')
        axes[1].axvline(tumor_center[0], color='green', linestyle=':', linewidth=2, label='Центр опухоли')
        axes[1].set_xlabel('Позиция Y (пиксели)', fontsize=11)
        axes[1].set_ylabel('Температура (°C)', fontsize=11)
        axes[1].set_title(f'Вертикальный срез через X={x_slice}', fontsize=12, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        # Срез через центр груди
        y_slice = h // 2
        x_range = np.arange(w)
        true_slice = temp_true[y_slice, :]
        recon_slice = temp_recon[y_slice, :]
        mask_slice = breast_mask[y_slice, :]
        
        axes[0].plot(x_range[mask_slice], true_slice[mask_slice], 'b-', linewidth=2, label='Истинная T')
        axes[0].plot(x_range[mask_slice], recon_slice[mask_slice], 'r--', linewidth=2, label='Реконструированная T')
        axes[0].set_xlabel('Позиция X (пиксели)', fontsize=11)
        axes[0].set_ylabel('Температура (°C)', fontsize=11)
        axes[0].set_title(f'Горизонтальный срез через Y={y_slice}', fontsize=12, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].axis('off')
        axes[1].text(0.5, 0.5, 'Опухоль не найдена\nСрез через центр груди', 
                    ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
    
    plt.tight_layout()
    plt.savefig('05_cross_section.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_emissivity_map(eps_map, breast_mask):
    """Карта коэффициента излучения ткани"""
    sqrt_eps = np.sqrt(np.maximum(eps_map, 1.0))
    gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
    emissivity = 1.0 - gamma**2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    im1 = axes[0].imshow(eps_map, cmap='viridis', vmin=0, vmax=60)
    axes[0].set_title('Диэлектрическая проницаемость (ε)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    plt.colorbar(im1, ax=axes[0], label='ε')
    
    emissivity_display = emissivity.copy()
    emissivity_display[~breast_mask] = np.nan
    im2 = axes[1].imshow(emissivity_display, cmap='plasma', vmin=0.5, vmax=1.0)
    axes[1].set_title('Коэффициент излучения (Emissivity)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    plt.colorbar(im2, ax=axes[1], label='Emissivity')
    
    plt.tight_layout()
    plt.savefig('06_emissivity_map.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_full_statistics(temp_true, temp_recon, breast_mask, Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, model):
    """Полная статистика работы модели"""
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
        # Исправлено: используем ogrid для создания сетки координат
        y_coords, x_coords = np.ogrid[:temp_true.shape[0], :temp_true.shape[1]]
        tumor_region = (x_coords - tx)**2 + (y_coords - ty)**2 <= 10**2
        
        # Проверка, попала ли область опухоли в маску груди
        if np.sum(tumor_region & breast_mask) > 0:
            tumor_true = temp_true[tumor_region & breast_mask]
            tumor_recon = temp_recon[tumor_region & breast_mask]
            print(f"   Координаты:         Y={ty}, X={tx}")
            print(f"   T в опухоли (истина): {tumor_true.mean():.2f} °C")
            print(f"   T в опухоли (рекон):  {tumor_recon.mean():.2f} °C")
            print(f"   Контраст опухоли:     {tumor_true.mean() - valid_true.mean():.2f} °C")
        else:
            print("   Опухоль за пределами груди (ошибка генерации)")
    else:
        print("   Опухоль не была создана")
    
    print("\n" + "="*70)

# =============================================================================
# 🚀 ОСНОВНАЯ ПРОГРАММА
# =============================================================================

if __name__ == "__main__":
    # Настройка стиля графиков (исправлено на универсальный)
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')  # Запасной вариант, если стиль не найден
    
    print("="*70)
    print("🔬 МОДЕЛЬ РАДИОМЕТРИИ МОЛОЧНОЙ ЖЕЛЕЗЫ")
    print("="*70)
    
    # 1. Инициализация модели
    model = BreastRadiometryModelReal(freq_ghz=3.0, resolution_mm=4)
    
    # 2. Создание фантома
    print("\n📌 Генерация анатомического фантома...")
    eps_map, cond_map, temp_true, breast_mask = model.create_anatomical_phantom(
        shape=(120, 180), 
        tumor_radius=12
    )
    
    # 3. Настройка сканирования
    h, w = temp_true.shape
    scan_y = int(h * 0.35)
    x_pos = np.linspace(int(w * 0.2), int(w * 0.8), 18, dtype=int)
    scan_grid = [(scan_y, x) for x in x_pos]
    
    print(f"📡 Количество антенн: {len(scan_grid)}")
    print(f"📍 Позиция сканера: Y={scan_y}")
    
    # 4. Прямая задача
    print("\n📡 Выполнение прямого сканирования...")
    Tb_data, emissivity_avg = model.forward_scan(eps_map, cond_map, temp_true, breast_mask, scan_grid)
    Tb_noisy = Tb_data + np.random.normal(0, 0.10, size=Tb_data.shape)
    
    # 5. Обратная задача
    print("🔄 Реконструкция температуры...")
    temp_recon = model.reconstruct_simple(Tb_noisy, emissivity_avg, scan_grid, temp_true.shape, breast_mask)
    
    # 6. Визуализация
    print("\n📊 Генерация графиков...")
    
    plot_main_results(temp_true, temp_recon, breast_mask, model.tumor_center)
    plot_sensitivity_kernels(model, breast_mask, scan_grid, n_show=5)
    plot_measurement_data(Tb_data, Tb_noisy, emissivity_avg, scan_grid)
    plot_temperature_histogram(temp_true, temp_recon, breast_mask)
    plot_cross_section(temp_true, temp_recon, breast_mask, model.tumor_center)
    plot_emissivity_map(eps_map, breast_mask)
    
    # 7. Статистика (передаем все необходимые переменные)
    print_full_statistics(temp_true, temp_recon, breast_mask, Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, model)
    
    print("\n✅ Все графики сохранены в файлы 01_*.png ... 06_*.png")
    print("="*70)