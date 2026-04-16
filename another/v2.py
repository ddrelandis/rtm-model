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
                'temp': 35.5
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
        
        # Нормализация к физическому диапазону температур
        valid_data = recon_field[mask]
        if len(valid_data) > 0:
            min_t, max_t = np.percentile(valid_data, [5, 95])
            if max_t > min_t:
                recon_field = 35.0 + (recon_field - min_t) / (max_t - min_t) * 4.0
        
        recon_field[~mask] = np.nan
        return recon_field

# --- Пример использования ---
if __name__ == "__main__":
    model = BreastRadiometryModelReal(freq_ghz=3.0, resolution_mm=4)
    
    print("Генерация анатомического фантома на основе данных IT'IS...")
    eps, cond, temp_true, breast_mask = model.create_anatomical_phantom(shape=(80, 100), tumor_radius=10)
    
    h, w = temp_true.shape
    scan_y = int(h * 0.45)
    x_pos = np.linspace(int(w * 0.2), int(w * 0.8), 5, dtype=int)
    scan_grid = [(scan_y, x) for x in x_pos]
    
    Tb_data, emissivity_avg = model.forward_scan(eps, cond, temp_true, breast_mask, scan_grid)
    Tb_noisy = Tb_data + np.random.normal(0, 0.15, size=Tb_data.shape)
    
    print(f"Проведено измерений: {len(Tb_noisy)}")
    print(f"Диапазон Tb: {Tb_noisy.min():.2f} - {Tb_noisy.max():.2f} K")
    print(f"Средний emissivity: {emissivity_avg.mean():.3f} ± {emissivity_avg.std():.3f}")
    
    temp_recon = model.reconstruct_simple(Tb_noisy, emissivity_avg, scan_grid, temp_true.shape, breast_mask)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    temp_display = temp_true.copy()
    temp_display[~breast_mask] = np.nan
    im1 = axes[0].imshow(temp_display, cmap='jet', vmin=34, vmax=39)
    axes[0].set_title('Истинное распределение T')
    plt.colorbar(im1, ax=axes[0], label='Температура (°C)')
    
    im2 = axes[1].imshow(temp_recon, cmap='jet', vmin=34, vmax=39)
    axes[1].set_title('Реконструированное T')
    plt.colorbar(im2, ax=axes[1], label='Температура (°C)')
    
    diff = np.abs(temp_true - temp_recon)
    diff[~breast_mask] = np.nan
    im3 = axes[2].imshow(diff, cmap='magma', vmin=0, vmax=2)
    axes[2].set_title('Абсолютная ошибка реконструкции')
    plt.colorbar(im3, ax=axes[2], label='Ошибка (°C)')
    
    plt.tight_layout()
    plt.show()
    
    print("\n--- Статистика диэлектрических свойств (внутри груди) ---")
    print(f"EPS (Среднее): {np.mean(eps[breast_mask]):.2f} ± {np.std(eps[breast_mask]):.2f}")
    print(f"COND (Среднее): {np.mean(cond[breast_mask]):.2f} ± {np.std(cond[breast_mask]):.2f} См/м")
    
    print("\n--- Статистика реконструкции ---")
    valid_recon = temp_recon[breast_mask]
    valid_true = temp_true[breast_mask]
    print(f"Истинная T: {valid_true.mean():.2f} ± {valid_true.std():.2f} °C")
    print(f"Реконструированная T: {valid_recon.mean():.2f} ± {valid_recon.std():.2f} °C")
    print(f"Средняя ошибка: {np.mean(np.abs(valid_true - valid_recon)):.2f} °C")