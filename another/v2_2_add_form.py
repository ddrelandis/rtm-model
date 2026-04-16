import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, distance_transform_edt

# =============================================================================
# 🏗️ КЛАСС МОДЕЛИ
# =============================================================================

class BreastRadiometryModelReal:
    def __init__(self, freq_ghz=3.5, resolution_mm=2):
        self.freq = freq_ghz * 1e9
        self.c = 3e8
        self.lambda0 = self.c / self.freq
        self.res = resolution_mm / 1000.0
        self.tumor_center = None
        
        self.tissue_props = {
            'fat': {
                'mean_eps': 10.5, 'std_eps': 1.5, 
                'mean_cond': 0.15, 'std_cond': 0.05, 
                'temp_base': 34.5
            },
            'gland': {
                'mean_eps': 48.0, 'std_eps': 6.0, 
                'mean_cond': 2.6, 'std_cond': 0.4, 
                'temp_base': 35.0
            },
            'tumor': {
                'mean_eps': 58.0, 'std_eps': 8.0, 
                'mean_cond': 4.2, 'std_cond': 0.8, 
                'temp_base': 37.0
            },
            'nipple': {
                'mean_eps': 52.0, 'std_eps': 5.0, 
                'mean_cond': 3.0, 'std_cond': 0.5, 
                'temp_base': 35.5
            },
            'body': {
                'mean_eps': 50.0, 'std_eps': 5.0, 
                'mean_cond': 2.0, 'std_cond': 0.3, 
                'temp_base': 37.0
            }
        }

    def get_tissue_values(self, tissue_type, shape):
        props = self.tissue_props[tissue_type]
        eps_map = np.clip(np.random.normal(props['mean_eps'], props['std_eps'], shape), 1.0, None)
        cond_map = np.clip(np.random.normal(props['mean_cond'], props['std_cond'], shape), 0.01, None)
        temp_map = np.ones(shape) * props['temp_base']
        return eps_map, cond_map, temp_map

    def create_anatomical_phantom(self, shape=(60, 80), tumor_radius=8, tumor_pos=None):
        """
        Создание анатомического фантома с РЕАЛЬНО ПЛАВНЫМИ температурными переходами.
        """
        h, w = shape
        eps_map = np.zeros(shape)
        cond_map = np.zeros(shape)
        temp_map = np.zeros(shape)
        
        y, x = np.ogrid[:h, :w]
        center_x = w / 2.0
        
        # =========================================================
        # 🎨 1. ФОРМА ГРУДИ
        # =========================================================
        
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
        
        breast_mask = binary_dilation(breast_mask, iterations=1)
        breast_mask = gaussian_filter(breast_mask.astype(float), sigma=0.8) > 0.5
        
        # =========================================================
        # 🎯 2. СОСОК
        # =========================================================
        
        nipple_center_y = int(h * 0.15)
        nipple_center_x = int(w / 2.0)
        areola_radius = int(w * 0.10)
        nipple_radius = int(w * 0.04)
        
        areola_mask = (x - nipple_center_x)**2 + (y - nipple_center_y)**2 <= areola_radius**2
        areola_mask = areola_mask & breast_mask
        
        nipple_mask = (x - nipple_center_x)**2 + (y - nipple_center_y)**2 <= nipple_radius**2
        nipple_mask = nipple_mask & areola_mask
        
        # =========================================================
        # 🧊 3. РАСПРЕДЕЛЕНИЕ ТКАНЕЙ
        # =========================================================
        
        body_transition_y = int(h * 0.70)
        body_mask = (y >= body_transition_y) & breast_mask
        main_breast_mask = breast_mask & ~areola_mask & (y < body_transition_y)
        
        if np.any(main_breast_mask):
            eps_map[main_breast_mask], cond_map[main_breast_mask], temp_map[main_breast_mask] = self.get_tissue_values('fat', np.sum(main_breast_mask))
        
        if np.any(body_mask):
            eps_map[body_mask], cond_map[body_mask], temp_map[body_mask] = self.get_tissue_values('body', np.sum(body_mask))
        
        eps_map[~breast_mask] = 1.0
        cond_map[~breast_mask] = 0.0
        temp_map[~breast_mask] = 20.0
        
        if np.any(areola_mask):
            e, c, t = self.get_tissue_values('gland', np.sum(areola_mask))
            eps_map[areola_mask] = e
            cond_map[areola_mask] = c
            temp_map[areola_mask] = t
        
        if np.any(nipple_mask):
            e, c, t = self.get_tissue_values('nipple', np.sum(nipple_mask))
            eps_map[nipple_mask] = e
            cond_map[nipple_mask] = c
            temp_map[nipple_mask] = t
        
        # =========================================================
        # 🔬 4. ФИБРОГЛАНДУЛЯРНАЯ ТКАНЬ
        # =========================================================
        
        n_clusters = 50
        gland_indices = np.where(main_breast_mask)
        for _ in range(n_clusters):
            if len(gland_indices[0]) == 0:
                break
            idx = np.random.randint(0, len(gland_indices[0]))
            cy, cx = gland_indices[0][idx], gland_indices[1][idx]
            cluster_size = np.random.randint(4, 12)
            yy, xx = np.ogrid[:h, :w]
            cluster_mask = (xx - cx)**2 + (yy - cy)**2 <= cluster_size**2
            region_mask = cluster_mask & main_breast_mask
            if np.any(region_mask):
                e, c, t = self.get_tissue_values('gland', np.sum(region_mask))
                eps_map[region_mask] = e
                cond_map[region_mask] = c
                temp_map[region_mask] = t
        
        # =========================================================
        # 🌡️ 5. ПЛАВНЫЙ ТЕМПЕРАТУРНЫЙ ГРАДИЕНТ (ИСПРАВЛЕНО!)
        # =========================================================
        # Ключевое изменение: создаём непрерывное 2D поле температуры
        
        # 5.1 Расстояние от поверхности груди (для градиента глубины)
        # Используем distance transform от границы груди
        breast_boundary = binary_erosion(breast_mask, iterations=1) ^ breast_mask
        dist_from_surface = distance_transform_edt(~breast_boundary)
        dist_from_surface = dist_from_surface.astype(float)
        dist_from_surface[~breast_mask] = 0
        
        # Нормализуем расстояние (0 = поверхность, 1 = максимальная глубина)
        max_dist = dist_from_surface[breast_mask].max()
        if max_dist > 0:
            normalized_depth = dist_from_surface / max_dist
        else:
            normalized_depth = np.zeros_like(dist_from_surface)
        
        # 5.2 Создаём плавный градиент температуры от поверхности к глубине
        # Температура растёт нелинейно (быстрее у поверхности, медлее в глубине)
        depth_temp = 34.5 + 2.5 * (normalized_depth ** 0.6)  # 34.5°C → 37.0°C
        
        # 5.3 Добавляем радиальный градиент от центра груди (центр теплее краёв)
        center_y = int(h * 0.50)
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist_center = np.sqrt((w/2)**2 + (h/2)**2)
        radial_factor = 1.0 - 0.15 * (dist_from_center / max_dist_center)  # Центр на 15% теплее
        
        # 5.4 Комбинируем градиенты
        temp_field = depth_temp * radial_factor
        
        # 5.5 Применяем к тканям (только внутри груди)
        temp_map = np.where(breast_mask, temp_field, temp_map)
        
        # 5.6 Добавляем случайные вариации (гетерогенность тканей)
        noise = np.random.normal(0, 0.15, shape)
        temp_map = temp_map + noise * breast_mask
        
        # 5.7 МНОГОКРАТНОЕ СГЛАЖИВАНИЕ (ключ к плавности!)
        # Применяем гауссов фильтр к ВСЕМУ массиву, не только к masked значениям
        temp_map = gaussian_filter(temp_map, sigma=4.0)
        
        # 5.8 Восстанавливаем фон после сглаживания
        temp_map[~breast_mask] = 20.0
        
        # 5.9 Ещё одно сглаживание для устранения артефактов на границе
        temp_smooth = gaussian_filter(temp_map, sigma=2.0)
        temp_map = np.where(breast_mask, temp_smooth, temp_map)
        
        # 5.10 Ограничиваем диапазон
        temp_map = np.clip(temp_map, 33.5, 38.5)
        temp_map[~breast_mask] = 20.0
        
        # =========================================================
        # 🎗️ 6. ОПУХОЛЬ С ОЧЕНЬ ПЛАВНЫМИ ГРАНИЦАМИ
        # =========================================================
        
        self.tumor_center = None
        
        if tumor_pos is not None:
            ty, tx = tumor_pos
            if 0 <= ty < h and 0 <= tx < w:
                if main_breast_mask[ty, tx]:
                    print(f"✅ Опухоль создана в заданной позиции: Y={ty}, X={tx}")
                else:
                    print(f"⚠️ Позиция ({ty}, {tx}) вне допустимой области! Коррекция...")
                    y_coords, x_coords = np.where(main_breast_mask)
                    if len(y_coords) > 0:
                        dists = np.sqrt((y_coords - ty)**2 + (x_coords - tx)**2)
                        nearest_idx = np.argmin(dists)
                        ty, tx = y_coords[nearest_idx], x_coords[nearest_idx]
                        print(f"✅ Скорректированная позиция: Y={ty}, X={tx}")
                    else:
                        print("⚠️ Не удалось скорректировать позицию")
                        tumor_pos = None
            else:
                print(f"⚠️ Позиция ({ty}, {tx}) вне сетки! Генерация случайной...")
                tumor_pos = None
        
        if tumor_pos is None:
            valid_y, valid_x = np.where(main_breast_mask & (y > h*0.30) & (y < h*0.65))
            if len(valid_y) > 0:
                idx = np.random.randint(0, len(valid_y))
                ty, tx = valid_y[idx], valid_x[idx]
                print(f"🎲 Опухоль создана в случайной позиции: Y={ty}, X={tx}")
            else:
                print("⚠️ Не удалось найти позицию для опухоли")
                return eps_map, cond_map, temp_map, breast_mask, areola_mask, nipple_mask, body_mask
        
        # Опухоль с ОЧЕНЬ плавным распределением температуры (широкий гаусс)
        tumor_y, tumor_x = np.ogrid[:h, :w]
        dist_from_tumor = np.sqrt((tumor_x - tx)**2 + (tumor_y - ty)**2)
        
        # Широкий гаусс для плавного перехода (sigma = 2.5 * radius)
        tumor_sigma = tumor_radius * 2.5
        tumor_temp_elevation = 2.0 * np.exp(-dist_from_tumor**2 / (2 * tumor_sigma**2))
        
        # Добавляем к базовой температуре
        temp_map = temp_map + tumor_temp_elevation * breast_mask
        temp_map = np.clip(temp_map, 33.5, 39.5)
        
        # Финальное сглаживание опухоли вместе с окружающими тканями
        temp_map = gaussian_filter(temp_map, sigma=2.5)
        temp_map[~breast_mask] = 20.0
        
        # Обновляем диэлектрические свойства опухоли (также плавно)
        tumor_eps_elevation = 15.0 * np.exp(-dist_from_tumor**2 / (2 * tumor_sigma**2))
        eps_map = eps_map + tumor_eps_elevation * breast_mask
        eps_map = gaussian_filter(eps_map, sigma=2.0)
        eps_map[~breast_mask] = 1.0
        
        self.tumor_center = (ty, tx)

        return eps_map, cond_map, temp_map, breast_mask, areola_mask, nipple_mask, body_mask

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

def plot_main_results(temp_true, temp_recon, breast_mask, tumor_center=None, areola_mask=None, nipple_mask=None, body_mask=None):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    temp_display = temp_true.copy()
    temp_display[~breast_mask] = np.nan
    im1 = axes[0].imshow(temp_display, cmap='jet', vmin=34, vmax=39, interpolation='gaussian')
    axes[0].set_title('Истинное распределение T', fontsize=12, fontweight='bold')
    
    if nipple_mask is not None:
        axes[0].contour(nipple_mask, colors='darkred', linewidths=3, alpha=0.9)
    if areola_mask is not None:
        axes[0].contour(areola_mask, colors='coral', linewidths=2, alpha=0.7)
    
    if tumor_center:
        axes[0].plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, markeredgewidth=2, label='Опухоль')
        axes[0].legend(loc='lower right')
    plt.colorbar(im1, ax=axes[0], label='Температура (°C)')
    
    im2 = axes[1].imshow(temp_recon, cmap='jet', vmin=34, vmax=39, interpolation='gaussian')
    axes[1].set_title('Реконструированное T', fontsize=12, fontweight='bold')
    if tumor_center:
        axes[1].plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, markeredgewidth=2)
    plt.colorbar(im2, ax=axes[1], label='Температура (°C)')
    
    diff = np.abs(temp_true - temp_recon)
    diff[~breast_mask] = np.nan
    im3 = axes[2].imshow(diff, cmap='magma', vmin=0, vmax=2, interpolation='gaussian')
    axes[2].set_title('Абсолютная ошибка', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], label='Ошибка (°C)')
    
    plt.tight_layout()
    plt.savefig('01_main_results.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_breast_anatomy(eps_map, breast_mask, areola_mask, nipple_mask, body_mask):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    eps_display = eps_map.copy()
    eps_display[~breast_mask] = np.nan
    im1 = axes[0].imshow(eps_display, cmap='viridis', vmin=0, vmax=60, interpolation='gaussian')
    axes[0].set_title('Диэлектрическая проницаемость (ε)', fontsize=12, fontweight='bold')
    
    if nipple_mask is not None:
        axes[0].contour(nipple_mask, colors='darkred', linewidths=3, alpha=0.9)
    if areola_mask is not None:
        axes[0].contour(areola_mask, colors='coral', linewidths=2, alpha=0.7)
    
    plt.colorbar(im1, ax=axes[0], label='ε')
    
    anatomy = np.zeros(breast_mask.shape)
    anatomy[breast_mask] = 1
    anatomy[areola_mask] = 2
    anatomy[nipple_mask] = 3
    anatomy[body_mask] = 4
    
    im2 = axes[1].imshow(anatomy, cmap='tab10', vmin=0, vmax=4)
    axes[1].set_title('Анатомическая структура', fontsize=12, fontweight='bold')
    
    legend_text = '1 - Жировая ткань\n2 - Ареола\n3 - Сосок\n4 - Тело (грудная стенка)'
    axes[1].text(0.02, 0.95, legend_text, transform=axes[1].transAxes, fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.colorbar(im2, ax=axes[1], label='Тип ткани')
    
    plt.tight_layout()
    plt.savefig('07_breast_anatomy.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_temperature_gradient(temp_map, breast_mask, tumor_center=None):
    """Визуализация температурного градиента по глубине"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    h, w = temp_map.shape
    
    center_x = w // 2
    temp_profile = temp_map[:, center_x].copy()
    temp_profile[~breast_mask[:, center_x]] = np.nan
    
    y_coords = np.arange(h)
    valid_mask = breast_mask[:, center_x]
    
    axes[0].plot(temp_profile[valid_mask], y_coords[valid_mask], 'b-', linewidth=2.5)
    axes[0].fill_betweenx(y_coords[valid_mask], temp_profile[valid_mask], 34, alpha=0.3)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Температура (°C)', fontsize=11)
    axes[0].set_ylabel('Глубина (пиксели)', fontsize=11)
    axes[0].set_title('Температурный градиент по глубине', fontsize=12, fontweight='bold')
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
    axes[1].set_title('Распределение температур в тканях', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('08_temperature_gradient.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_temperature_contours(temp_map, breast_mask, tumor_center=None):
    """Показывает изотермы для визуализации плавности градиента"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    temp_display = temp_map.copy()
    temp_display[~breast_mask] = np.nan
    
    im = ax.imshow(temp_display, cmap='jet', vmin=34, vmax=39, interpolation='gaussian')
    
    # Добавляем контуры температур (изотермы)
    contour_levels = np.arange(34.5, 39.0, 0.5)
    cs = ax.contour(temp_display, levels=contour_levels, colors='white', linewidths=1.0, alpha=0.8)
    ax.clabel(cs, inline=True, fontsize=8, fmt='%.1f°C')
    
    if tumor_center:
        ax.plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, markeredgewidth=2, label='Опухоль')
        ax.legend(loc='lower right')
    
    ax.set_title('Изотермы температуры (плавность градиента)', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Температура (°C)')
    
    plt.tight_layout()
    plt.savefig('09_temperature_contours.png', dpi=150, bbox_inches='tight')
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
    axes[0].set_title('Распределение истинных температур', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
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
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    h, w = temp_true.shape
    
    if tumor_center:
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
    
    im1 = axes[0].imshow(eps_map, cmap='viridis', vmin=0, vmax=60, interpolation='gaussian')
    axes[0].set_title('Диэлектрическая проницаемость (ε)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='ε')
    
    emissivity_display = emissivity.copy()
    emissivity_display[~breast_mask] = np.nan
    im2 = axes[1].imshow(emissivity_display, cmap='plasma', vmin=0.5, vmax=1.0, interpolation='gaussian')
    axes[1].set_title('Коэффициент излучения (Emissivity)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Emissivity')
    
    plt.tight_layout()
    plt.savefig('06_emissivity_map.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_full_statistics(temp_true, temp_recon, breast_mask, Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, model):
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
    
    # ✅ ИСПРАВЛЕНО: Расчёт градиента на 2D массиве с маской
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
    print("🔬 МОДЕЛЬ РАДИОМЕТРИИ МОЛОЧНОЙ ЖЕЛЕЗЫ (ИДЕАЛЬНО ПЛАВНЫЕ ГРАДИЕНТЫ)")
    print("="*70)
    
    model = BreastRadiometryModelReal(freq_ghz=3.0, resolution_mm=4)
    
    print("\n📌 Генерация анатомического фантома...")
    
    tumor_position = (45, 50)
    
    eps_map, cond_map, temp_true, breast_mask, areola_mask, nipple_mask, body_mask = model.create_anatomical_phantom(
        shape=(80, 100), 
        tumor_radius=8,
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
    
    plot_main_results(temp_true, temp_recon, breast_mask, model.tumor_center, areola_mask, nipple_mask, body_mask)
    plot_breast_anatomy(eps_map, breast_mask, areola_mask, nipple_mask, body_mask)
    plot_temperature_gradient(temp_true, breast_mask, model.tumor_center)
    plot_temperature_contours(temp_true, breast_mask, model.tumor_center)  # НОВЫЙ ГРАФИК
    plot_sensitivity_kernels(model, breast_mask, scan_grid, n_show=5)
    plot_measurement_data(Tb_data, Tb_noisy, emissivity_avg, scan_grid)
    plot_temperature_histogram(temp_true, temp_recon, breast_mask)
    plot_cross_section(temp_true, temp_recon, breast_mask, model.tumor_center)
    plot_emissivity_map(eps_map, breast_mask)
    
    print_full_statistics(temp_true, temp_recon, breast_mask, Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, model)
    
    print("\n✅ Все графики сохранены в файлы 01_*.png ... 09_*.png")
    print("="*70)