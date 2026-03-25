import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, distance_transform_edt
import time

# =============================================================================
# 🏗️ 3D КЛАСС МОДЕЛИ
# =============================================================================
class BreastRadiometryModel3D:
    def __init__(self, freq_ghz=3.0, resolution_mm=2, birads_category='B', temp_vmin=None, temp_vmax=None):
        self.freq = freq_ghz * 1e9
        self.c = 3e8
        self.lambda0 = self.c / self.freq
        self.res = resolution_mm / 1000.0
        self.tumor_center = None  # Теперь (z, y, x)
        self.birads_category = birads_category
        self.birads_density = {
            'A': (0.10, 0.25),
            'B': (0.26, 0.50),
            'C': (0.51, 0.75),
            'D': (0.76, 0.90)
        }
        self.temp_vmin = temp_vmin if temp_vmin is not None else 33.0
        self.temp_vmax = temp_vmax if temp_vmax is not None else 40.0
        self.tissue_props = {
            'fat': {'mean_eps': 5.0, 'std_eps': 0.5, 'mean_cond': 0.10, 'std_cond': 0.03, 'temp_base': 35.0, 'temp_offset': 0.0},
            'fat_subcutaneous': {'mean_eps': 4.5, 'std_eps': 0.4, 'mean_cond': 0.08, 'std_cond': 0.02, 'temp_base': 34.8, 'temp_offset': 0.0},
            'fat_retromammary': {'mean_eps': 5.0, 'std_eps': 0.5, 'mean_cond': 0.10, 'std_cond': 0.03, 'temp_base': 35.0, 'temp_offset': 0.0},
            'gland': {'mean_eps': 45.0, 'std_eps': 5.0, 'mean_cond': 2.4, 'std_cond': 0.4, 'temp_base': 35.0, 'temp_offset': 0.8},
            'gland_ducts': {'mean_eps': 48.0, 'std_eps': 5.0, 'mean_cond': 2.8, 'std_cond': 0.5, 'temp_base': 35.0, 'temp_offset': 1.0},
            'connective': {'mean_eps': 30.0, 'std_eps': 4.0, 'mean_cond': 1.3, 'std_cond': 0.3, 'temp_base': 35.0, 'temp_offset': 0.3},
            'tumor': {'mean_eps': 55.0, 'std_eps': 7.0, 'mean_cond': 4.0, 'std_cond': 0.8, 'temp_base': 38.0, 'temp_offset': 0.0},
            'nipple': {'mean_eps': 45.0, 'std_eps': 5.0, 'mean_cond': 2.6, 'std_cond': 0.5, 'temp_base': 35.0, 'temp_offset': 0.6},
            'body': {'mean_eps': 50.0, 'std_eps': 5.0, 'mean_cond': 2.0, 'std_cond': 0.3, 'temp_base': 35.0, 'temp_offset': 0.0},
            'skin': {'mean_eps': 35.0, 'std_eps': 4.0, 'mean_cond': 1.0, 'std_cond': 0.2, 'temp_base': 33.8, 'temp_offset': 0.0}
        }

    def get_tissue_values(self, tissue_type, size):
        props = self.tissue_props[tissue_type]
        eps_map = np.clip(np.random.normal(props['mean_eps'], props['std_eps'], size), 1.0, None)
        cond_map = np.clip(np.random.normal(props['mean_cond'], props['std_cond'], size), 0.01, None)
        temp_map = np.ones(size) * props['temp_base']
        return eps_map, cond_map, temp_map, props['temp_offset']

    def create_anatomical_phantom_realistic(self, shape=(80, 160, 200), tumor_radius=12, tumor_pos=None):
        """
        🔥 3D ФАНТОМ МОЛОЧНОЙ ЖЕЛЕЗЫ
        shape: (depth, height, width) = (Z, Y, X)
        """
        start_time = time.time()
        d, h, w = shape  # ✅ 3D: глубина, высота, ширина
        
        eps_map = np.ones(shape) * 1.0
        cond_map = np.zeros(shape)
        temp_map = np.zeros(shape) + 20.0
        tissue_type_map = np.zeros(shape, dtype=int)
        
        z, y, x = np.ogrid[:d, :h, :w]  # ✅ 3D координаты
        center_x = w * 0.45
        scale_factor = h / 80.0
        
        print(f"🔍 3D Разрешение: {d}×{h}×{w} (масштаб: {scale_factor:.2f}×)")
        
        # =====================================================================
        # 1. 3D ФОРМА ГРУДИ (эллипсоид)
        # =====================================================================
        breast_mask = np.zeros(shape, dtype=bool)
        
        for zi in range(d):  # ✅ Цикл по глубине Z
            depth_factor = 1.0 - (zi / d) ** 0.5  # Сужение к грудной стенке
            for yi in range(int(h * 0.12), h):  # ✅ Цикл по высоте Y
                normalized_y = (yi - h * 0.12) / (h * 0.88)
                if 0 <= normalized_y <= 1:
                    width_factor = w * 0.35 * depth_factor * (1 + 0.3 * np.sin(normalized_y * np.pi))
                    x_left = int(center_x - width_factor)
                    x_right = int(center_x + width_factor * 0.8)
                    x_left = max(0, x_left)
                    x_right = min(w, x_right)
                    # ✅ 3D индексация: [z, y, x]
                    breast_mask[zi, yi, x_left:x_right] = True
        
        breast_mask = gaussian_filter(breast_mask.astype(float), sigma=1.0 * scale_factor) > 0.45
        breast_mask = binary_dilation(breast_mask, iterations=max(1, int(1.5 * scale_factor)))
        
        # =====================================================================
        # 2. 3D ТЕМПЕРАТУРНЫЙ ГРАДИЕНТ
        # =====================================================================
        dist_from_surface = distance_transform_edt(~breast_mask) * self.res
        dist_from_surface[~breast_mask] = 0
        max_dist = dist_from_surface[breast_mask].max()
        
        if max_dist > 0:
            normalized_depth = dist_from_surface / max_dist
            depth_gradient = 2.5 * (normalized_depth ** 0.6)
            temp_map = 35.0 + depth_gradient * breast_mask
        else:
            temp_map = 35.0 * breast_mask
        
        # =====================================================================
        # 3. 3D ОПУХОЛЬ (сфера)
        # =====================================================================
        self.tumor_center = None
        
        if tumor_pos is None:
            valid_z, valid_y, valid_x = np.where(
                breast_mask & (z > d*0.3) & (z < d*0.7)
            )
            if len(valid_z) > 0:
                idx = np.random.randint(0, len(valid_z))
                tumor_pos = (valid_z[idx], valid_y[idx], valid_x[idx])
        
        if tumor_pos is not None:
            # ✅ 3D расстояние (сфера)
            dist_from_tumor = np.sqrt(
                (x - tumor_pos[2])**2 + 
                (y - tumor_pos[1])**2 + 
                (z - tumor_pos[0])**2
            ) * self.res
            
            tumor_sigma = tumor_radius * self.res
            tumor_temp = 3.2 * np.exp(-dist_from_tumor**2 / (2 * tumor_sigma**2))
            temp_map = temp_map + tumor_temp * breast_mask
            temp_map = np.clip(temp_map, 34.0, 39.5)
            
            tumor_eps = 18 * np.exp(-dist_from_tumor**2 / (2 * tumor_sigma**2))
            eps_map = eps_map + tumor_eps * breast_mask
            
            self.tumor_center = tumor_pos
            print(f"✅ Опухоль 3D: Z={tumor_pos[0]}, Y={tumor_pos[1]}, X={tumor_pos[2]}")
        
        # =====================================================================
        # 4. ЗАПОЛНЕНИЕ ДИЭЛЕКТРИЧЕСКИМИ СВОЙСТВАМИ
        # =====================================================================
        eps_map[breast_mask] = np.clip(np.random.normal(35.0, 15.0, np.sum(breast_mask)), 5.0, 60.0)
        cond_map[breast_mask] = np.clip(np.random.normal(1.5, 0.8, np.sum(breast_mask)), 0.1, 4.0)
        eps_map[~breast_mask] = 1.0
        cond_map[~breast_mask] = 0.0
        temp_map[~breast_mask] = 20.0
        
        temp_map = gaussian_filter(temp_map, sigma=1.0 * scale_factor)
        temp_map = np.clip(temp_map, 34.0, 39.5)
        temp_map[~breast_mask] = 20.0
        
        elapsed = time.time() - start_time
        print(f"⏱️ Время создания 3D фантома: {elapsed:.2f} сек")
        
        return eps_map, cond_map, temp_map, breast_mask, tissue_type_map

    def compute_sensitivity_kernel_3d(self, mask, ant_pos):
        """
        🔥 3D ЯДРО: сферическое затухание
        ant_pos: (z, y, x)
        """
        d, h, w = mask.shape
        z, y, x = np.ogrid[:d, :h, :w]
        
        dist = np.sqrt(
            (x - ant_pos[2])**2 + 
            (y - ant_pos[1])**2 + 
            (z - ant_pos[0])**2
        ) * self.res
        
        delta_eff = 0.05  # 5 см для 3 ГГц
        sensitivity = np.exp(-dist**2 / (2 * delta_eff**2)) * mask
        
        depth_weight = np.clip(1.0 + (z - ant_pos[0]) / 30.0, 0.5, 2.0)
        sensitivity = sensitivity * depth_weight
        
        sum_sens = np.sum(sensitivity)
        if sum_sens > 0:
            return sensitivity / sum_sens
        return sensitivity

    def compute_emissivity(self, eps_map, mask=None):
        sqrt_eps = np.sqrt(np.maximum(eps_map, 1.0))
        gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
        emissivity_fresnel = 1.0 - gamma**2
        emissivity = 0.88 + 0.11 * (emissivity_fresnel - 0.5) / 0.5
        emissivity = np.clip(emissivity, 0.90, 0.99)
        
        if mask is not None:
            np.random.seed(42)
            noise = np.random.normal(0, 0.015, emissivity.shape)
            emissivity = emissivity + noise * mask
            emissivity = np.clip(emissivity, 0.90, 0.99)
        
        return emissivity

    def forward_scan_3d(self, eps_map, cond_map, temp_map, mask, scan_positions):
        """
        🔥 3D СКАНИРОВАНИЕ
        """
        measurements = []
        emissivity_avg = []
        
        temp_map_kelvin = temp_map + 273.15
        emissivity = self.compute_emissivity(eps_map, mask)
        
        for pos in scan_positions:
            kernel = self.compute_sensitivity_kernel_3d(mask, pos)
            emissivity_local = np.sum(kernel * emissivity)
            emissivity_avg.append(emissivity_local)
            Tb = np.sum(kernel * emissivity * temp_map_kelvin)
            measurements.append(Tb)
        
        return np.array(measurements), np.array(emissivity_avg)

    def reconstruct_3d(self, measurements, emissivity_avg, scan_positions, shape, mask):
        """
        🔥 3D РЕКОНСТРУКЦИЯ
        """
        recon_field_kelvin = np.zeros(shape)
        weight_sum = np.zeros(shape)
        
        for i, pos in enumerate(scan_positions):
            kernel = self.compute_sensitivity_kernel_3d(mask, pos)
            emissivity_corr = emissivity_avg[i] if emissivity_avg[i] > 0.5 else 0.95
            Tb_corrected = measurements[i] / emissivity_corr
            recon_field_kelvin += kernel * Tb_corrected
            weight_sum += kernel
        
        weight_sum[weight_sum == 0] = 1.0
        recon_field_kelvin /= weight_sum
        recon_field = recon_field_kelvin - 273.15
        recon_field = gaussian_filter(recon_field, sigma=2.0)
        recon_field[~mask] = np.nan
        
        return recon_field

# =============================================================================
# 📊 3D ВИЗУАЛИЗАЦИЯ
# =============================================================================
def plot_3d_volume(temp_map, breast_mask, tumor_center=None):
    """
    🔥 3D ВИЗУАЛИЗАЦИЯ: срезы + проекции
    """
    fig = plt.figure(figsize=(20, 5))
    d, h, w = temp_map.shape
    
    # --- 3 Среза (XY, XZ, YZ) ---
    slices = [d//2, h//2, w//2]
    slice_names = ['XY (горизонт)', 'XZ (фронталь)', 'YZ (сагитталь)']
    slice_dims = [(0, slices[0]), (1, slices[1]), (2, slices[2])]
    
    for i, (name, (axis, idx)) in enumerate(zip(slice_names, slice_dims)):
        ax = fig.add_subplot(1, 4, i+1)
        
        if axis == 0:
            slice_data = temp_map[idx, :, :].copy()
            slice_mask = breast_mask[idx, :, :]
        elif axis == 1:
            slice_data = temp_map[:, idx, :].copy()
            slice_mask = breast_mask[:, idx, :]
        else:
            slice_data = temp_map[:, :, idx].copy()
            slice_mask = breast_mask[:, :, idx]
        
        slice_data[~slice_mask] = np.nan
        
        im = ax.imshow(slice_data, cmap='jet', vmin=34, vmax=40)
        ax.set_title(f'{name}\nСрез {idx}')
        plt.colorbar(im, ax=ax, label='T (°C)')
        
        if tumor_center and i == 0:
            if tumor_center[0] == idx:
                ax.plot(tumor_center[2], tumor_center[1], 'r+', markersize=15, label='Опухоль')
                ax.legend()
    
    # --- Гистограмма температур ---
    ax = fig.add_subplot(1, 4, 4)
    valid_temps = temp_map[breast_mask]
    ax.hist(valid_temps, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(valid_temps.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {valid_temps.mean():.2f}°C')
    ax.set_xlabel('Температура (°C)')
    ax.set_ylabel('Количество вокселей')
    ax.set_title('Распределение температур 3D')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('01_3d_volume.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_3d_slices(temp_true, temp_recon, breast_mask, tumor_center=None):
    """
    🔥 Сравнение истинной и реконструированной 3D температуры
    """
    d, h, w = temp_true.shape
    slices = [d//4, d//2, 3*d//4]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    for i, z_idx in enumerate(slices):
        true_slice = temp_true[z_idx, :, :].copy()
        true_slice[~breast_mask[z_idx, :, :]] = np.nan
        
        recon_slice = temp_recon[z_idx, :, :].copy()
        recon_slice[~breast_mask[z_idx, :, :]] = np.nan
        
        diff_slice = np.abs(true_slice - recon_slice)
        diff_slice[~breast_mask[z_idx, :, :]] = np.nan
        
        im1 = axes[i, 0].imshow(true_slice, cmap='jet', vmin=34, vmax=40)
        axes[i, 0].set_title(f'Истинная T\nZ={z_idx}')
        plt.colorbar(im1, ax=axes[i, 0])
        
        im2 = axes[i, 1].imshow(recon_slice, cmap='jet', vmin=34, vmax=40)
        axes[i, 1].set_title(f'Реконструированная T\nZ={z_idx}')
        plt.colorbar(im2, ax=axes[i, 1])
        
        im3 = axes[i, 2].imshow(diff_slice, cmap='magma', vmin=0, vmax=3)
        axes[i, 2].set_title(f'Ошибка\nZ={z_idx}')
        plt.colorbar(im3, ax=axes[i, 2])
        
        if tumor_center and tumor_center[0] == z_idx:
            axes[i, 0].plot(tumor_center[2], tumor_center[1], 'r+', markersize=15)
            axes[i, 1].plot(tumor_center[2], tumor_center[1], 'r+', markersize=15)
    
    plt.tight_layout()
    plt.savefig('02_3d_slices.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_3d_measurement_data(Tb_data, emissivity_avg, scan_positions):
    """
    🔥 Визуализация 3D измерений
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Группировка по плоскостям Z
    z_planes = {}
    for i, pos in enumerate(scan_positions):
        z = pos[0]
        if z not in z_planes:
            z_planes[z] = {'x': [], 'tb': [], 'idx': []}
        z_planes[z]['x'].append(pos[2])
        z_planes[z]['tb'].append(Tb_data[i])
        z_planes[z]['idx'].append(i)
    
    # График Tb по позициям
    colors = plt.cm.viridis(np.linspace(0, 1, len(z_planes)))
    for i, (z, data) in enumerate(sorted(z_planes.items())):
        axes[0].scatter(data['x'], data['tb'], c=[colors[i]], s=50, label=f'Z={z}', alpha=0.7)
    
    axes[0].set_xlabel('Позиция антенны (X)')
    axes[0].set_ylabel('Яркостная температура Tb (K)')
    axes[0].set_title('3D измерения яркостной температуры')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Emissivity
    axes[1].plot(range(len(emissivity_avg)), emissivity_avg, 'go-', linewidth=1, markersize=4)
    axes[1].axhline(y=emissivity_avg.mean(), color='r', linestyle='--', label=f'Среднее: {emissivity_avg.mean():.3f}')
    axes[1].set_xlabel('Индекс антенны')
    axes[1].set_ylabel('Emissivity')
    axes[1].set_title('Коэффициент излучения (3D)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('03_3d_measurements.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_3d_statistics(temp_true, temp_recon, breast_mask, Tb_data, emissivity_avg, eps_map, cond_map, model):
    """
    🔥 3D СТАТИСТИКА
    """
    print("\n" + "="*70)
    print("📊 3D СТАТИСТИКА РАБОТЫ МОДЕЛИ")
    print("="*70)
    
    valid_true = temp_true[breast_mask]
    valid_recon = temp_recon[breast_mask]
    
    print("\n📍 ДИЭЛЕКТРИЧЕСКИЕ СВОЙСТВА (3D):")
    print(f"   EPS:  {np.mean(eps_map[breast_mask]):6.2f} ± {np.std(eps_map[breast_mask]):.2f}")
    print(f"   COND: {np.mean(cond_map[breast_mask]):6.2f} ± {np.std(cond_map[breast_mask]):.2f} См/м")
    
    print("\n🌡️ ТЕМПЕРАТУРНАЯ СТАТИСТИКА (3D):")
    print(f"   Истинная T:        {valid_true.mean():6.2f} ± {valid_true.std():.2f} °C")
    print(f"   Реконструированная: {valid_recon.mean():6.2f} ± {valid_recon.std():.2f} °C")
    print(f"   Смещение (bias):    {valid_recon.mean() - valid_true.mean():+.2f} °C")
    print(f"   Мин T:              {valid_true.min():6.2f} °C")
    print(f"   Макс T:             {valid_true.max():6.2f} °C")
    
    print("\n📏 ОШИБКИ РЕКОНСТРУКЦИИ (3D):")
    abs_error = np.abs(valid_true - valid_recon)
    print(f"   Средняя (MAE):      {abs_error.mean():.2f} °C")
    print(f"   Максимальная:       {abs_error.max():.2f} °C")
    print(f"   RMSE:               {np.sqrt(np.mean(abs_error**2)):.2f} °C")
    
    print("\n📡 3D ИЗМЕРЕНИЯ РАДИОМЕТРА:")
    print(f"   Количество антенн:  {len(Tb_data)}")
    print(f"   Tb (мин):           {Tb_data.min():.2f} K ({Tb_data.min()-273.15:.2f}°C)")
    print(f"   Tb (макс):          {Tb_data.max():.2f} K ({Tb_data.max()-273.15:.2f}°C)")
    print(f"   Tb (среднее):       {Tb_data.mean():.2f} K ({Tb_data.mean()-273.15:.2f}°C)")
    print(f"   Emissivity (средн.): {emissivity_avg.mean():.3f} ± {emissivity_avg.std():.3f}")
    
    print("\n🎯 3D ДЕТЕКЦИЯ ОПУХОЛИ:")
    if model.tumor_center:
        tz, ty, tx = model.tumor_center
        tumor_region = (np.arange(temp_true.shape[0]).reshape(-1,1,1) - tz)**2 + \
                      (np.arange(temp_true.shape[1]).reshape(1,-1,1) - ty)**2 + \
                      (np.arange(temp_true.shape[2]).reshape(1,1,-1) - tx)**2 <= 10**2
        tumor_region = tumor_region & breast_mask
        if np.sum(tumor_region) > 0:
            tumor_true = temp_true[tumor_region]
            tumor_recon = temp_recon[tumor_region]
            print(f"   Координаты:         Z={tz}, Y={ty}, X={tx}")
            print(f"   T в опухоли (истина): {tumor_true.mean():.2f} °C")
            print(f"   T в опухоли (рекон):  {tumor_recon.mean():.2f} °C")
            print(f"   Контраст опухоли:     {tumor_true.mean() - valid_true.mean():.2f} °C")
    
    print("\n" + "="*70)

# =============================================================================
# 🚀 ОСНОВНАЯ ПРОГРАММА (3D)
# =============================================================================
if __name__ == "__main__":
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('default')
    
    print("="*70)
    print("🔬 3D МОДЕЛЬ РАДИОМЕТРИИ МОЛОЧНОЙ ЖЕЛЕЗЫ")
    print("="*70)
    
    # 🔥 3D ПАРАМЕТРЫ
    RESOLUTION_PRESETS_3D = {
        'low': {'shape': (40, 80, 100), 'tumor_radius': 8, 'resolution_mm': 4},
        'medium': {'shape': (80, 160, 200), 'tumor_radius': 12, 'resolution_mm': 2},
        'high': {'shape': (120, 240, 300), 'tumor_radius': 16, 'resolution_mm': 1}
    }
    
    quality = 'medium'
    preset = RESOLUTION_PRESETS_3D[quality]
    print(f"\n📐 3D Режим: {quality.upper()} ({preset['shape'][0]}×{preset['shape'][1]}×{preset['shape'][2]})")
    
    model = BreastRadiometryModel3D(
        freq_ghz=3.0,
        resolution_mm=preset['resolution_mm'],
        birads_category='B'
    )
    
    print("\n📌 Генерация 3D фантома...")
    start_total = time.time()
    
    eps_map, cond_map, temp_true, breast_mask, tissue_type_map = model.create_anatomical_phantom_realistic(
        shape=preset['shape'],
        tumor_radius=preset['tumor_radius']
    )
    
    d, h, w = preset['shape']
    
    # 🔥 3D СЕТКА АНТЕНН (3 плоскости × 25 позиций)
    scan_planes = [int(d * 0.25), int(d * 0.50), int(d * 0.75)]
    scan_grid_3d = []
    
    for z_pos in scan_planes:
        x_pos = np.linspace(int(w * 0.20), int(w * 0.80), 25, dtype=int)
        y_pos = int(h * 0.30)
        for x in x_pos:
            scan_grid_3d.append((z_pos, y_pos, x))
    
    print(f"\n📡 Количество антенн 3D: {len(scan_grid_3d)} (3 плоскости × 25)")
    
    # 🔥 3D СКАНИРОВАНИЕ
    print("\n📡 Выполнение 3D сканирования...")
    Tb_data, emissivity_avg = model.forward_scan_3d(
        eps_map, cond_map, temp_true, breast_mask, scan_grid_3d
    )
    
    print(f"\n🔍 3D Диагностика Tb:")
    print(f"   Мин Tb: {Tb_data.min():.2f} K ({Tb_data.min()-273.15:.2f}°C)")
    print(f"   Макс Tb: {Tb_data.max():.2f} K ({Tb_data.max()-273.15:.2f}°C)")
    print(f"   Среднее Tb: {Tb_data.mean():.2f} K ({Tb_data.mean()-273.15:.2f}°C)")
    
    if Tb_data.min() < 260 or Tb_data.max() > 320:
        print("   ⚠️ Tb вне диапазона (260-320 K)")
    else:
        print("   ✅ Tb в физиологическом диапазоне")
    
    # 🔥 3D РЕКОНСТРУКЦИЯ
    print("\n🔄 3D Реконструкция температуры...")
    temp_recon = model.reconstruct_3d(
        Tb_data, emissivity_avg, scan_grid_3d, preset['shape'], breast_mask
    )
    
    # 🔥 3D ВИЗУАЛИЗАЦИЯ
    print("\n📊 Генерация 3D графиков...")
    plot_3d_volume(temp_true, breast_mask, model.tumor_center)
    plot_3d_slices(temp_true, temp_recon, breast_mask, model.tumor_center)
    plot_3d_measurement_data(Tb_data, emissivity_avg, scan_grid_3d)
    print_3d_statistics(temp_true, temp_recon, breast_mask, Tb_data, emissivity_avg, eps_map, cond_map, model)
    
    total_elapsed = time.time() - start_total
    print(f"\n⏱️ Общее время 3D: {total_elapsed:.2f} сек ({total_elapsed/60:.1f} мин)")
    print("\n✅ 3D графики сохранены: 01_3d_*.png ... 03_3d_*.png")
    print("="*70)