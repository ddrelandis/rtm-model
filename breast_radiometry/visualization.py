import numpy as np
import matplotlib.pyplot as plt

def plot_main_results(temp_true, temp_recon, breast_mask, tumor_center=None,
                      areola_mask=None, nipple_mask=None, body_mask=None,
                      temp_vmin=34.0, temp_vmax=39.0):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax in axes: ax.set_facecolor('0.95')
    
    temp_display = temp_true.copy()
    temp_display[~breast_mask] = np.nan
    im1 = axes[0].imshow(temp_display, cmap='jet', vmin=temp_vmin, vmax=temp_vmax, interpolation='gaussian')
    axes[0].set_title(f'Истинное распределение T\n[{temp_vmin:.1f}°C — {temp_vmax:.1f}°C]', fontsize=12, fontweight='bold')
    if nipple_mask is not None: axes[0].contour(nipple_mask, colors='darkred', linewidths=3, alpha=0.9)
    if areola_mask is not None: axes[0].contour(areola_mask, colors='coral', linewidths=2, alpha=0.7)
    if tumor_center:
        axes[0].plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, markeredgewidth=2, label='Опухоль')
        axes[0].legend(loc='lower right')
    plt.colorbar(im1, ax=axes[0], label='Температура (°C)').set_ticks(np.linspace(temp_vmin, temp_vmax, 8))

    temp_recon_display = temp_recon.copy()
    temp_recon_display[~breast_mask] = np.nan
    im2 = axes[1].imshow(temp_recon_display, cmap='jet', vmin=temp_vmin, vmax=temp_vmax, interpolation='gaussian')
    axes[1].set_title(f'Реконструированное T\n[{temp_vmin:.1f}°C — {temp_vmax:.1f}°C]', fontsize=12, fontweight='bold')
    if tumor_center: axes[1].plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, markeredgewidth=2)
    plt.colorbar(im2, ax=axes[1], label='Температура (°C)').set_ticks(np.linspace(temp_vmin, temp_vmax, 8))
    recon_valid = temp_recon[breast_mask]
    if np.any(recon_valid < temp_vmin) or np.any(recon_valid > temp_vmax):
        print(f"ВНИМАНИЕ: Обнаружены значения вне диапазона! Мин: {np.nanmin(recon_valid):.2f}°C, Макс: {np.nanmax(recon_valid):.2f}°C")

    diff = np.abs(temp_true - temp_recon)
    diff[~breast_mask] = np.nan
    vmax_err = min(3.0, np.nanmax(diff))
    im3 = axes[2].imshow(diff, cmap='magma', vmin=0, vmax=vmax_err, interpolation='gaussian')
    axes[2].set_title(f'Абсолютная ошибка\n[0.0°C — {vmax_err:.1f}°C]', fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=axes[2], label='Ошибка (°C)').set_ticks(np.linspace(0, vmax_err, 5))
    
    plt.tight_layout()
    plt.savefig('01_main_results.png', dpi=150, bbox_inches='tight')
    plt.show()  #  графики без отображения
    print(f"\nДиапазоны температур:\n   Истинное T:     {np.min(temp_true[breast_mask]):.2f} — {np.max(temp_true[breast_mask]):.2f}°C\n   Реконструированное: {np.nanmin(temp_recon[breast_mask]):.2f} — {np.nanmax(temp_recon[breast_mask]):.2f}°C\n   Ошибка:         0.00 — {vmax_err:.2f}°C")

def plot_tissue_composition(tissue_type_map, breast_mask, areola_mask, nipple_mask, body_mask, birads_category):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    tissue_names = {0: 'Фон', 1: 'Подкожный жир', 2: 'Железистая', 3: 'Внутрижелез. жир', 4: 'Ретромаммарный', 5: 'Соединительная', 6: 'Протоки', 7: 'Дольки', 8: 'Сосок', 9: 'Ареола', 10: 'Кожа', 11: 'Тело'}
    colors = ['#000000', '#F9F0D9', '#D4A5A5', '#F4E4C1', '#E8D5A3', '#E0C0C0', '#C48585', '#B87070', '#8B4513', '#CD853F', '#FFE4C4', '#A0A0A0']
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    im1 = axes[0].imshow(tissue_type_map, cmap=cmap, vmin=0, vmax=11)
    axes[0].set_title(f'Гистологическая структура (BI-RADS {birads_category})', fontsize=12, fontweight='bold')
    legend_elements = [plt.Line2D([0], [0], marker='s', color='w', label=name, markerfacecolor=colors[i], markersize=10) for i, name in tissue_names.items() if i > 0]
    axes[0].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, framealpha=0.9)
    tissue_counts = {i: np.sum(tissue_type_map == i) / np.sum(breast_mask) * 100 for i in range(1, 12) if np.sum(tissue_type_map == i) > 0}
    axes[1].pie(tissue_counts.values(), labels=[tissue_names[i] for i in tissue_counts.keys()], colors=[colors[i] for i in tissue_counts.keys()], autopct='%1.1f%%', startangle=90, textprops={'fontsize': 8})
    axes[1].set_title('Распределение тканей (%)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('07_tissue_composition.png', dpi=150, bbox_inches='tight')
    plt.close()  

def plot_breast_anatomy(eps_map, breast_mask, areola_mask, nipple_mask, body_mask):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    eps_display = eps_map.copy()
    eps_display[~breast_mask] = np.nan
    vmin_eps, vmax_eps = np.min(eps_map[breast_mask]), np.max(eps_map[breast_mask])
    im1 = axes[0].imshow(eps_display, cmap='viridis', vmin=vmin_eps, vmax=vmax_eps, interpolation='gaussian')
    axes[0].set_title(f'Диэлектрическая проницаемость\n[{vmin_eps:.1f} — {vmax_eps:.1f}]', fontsize=12, fontweight='bold')
    if nipple_mask is not None: axes[0].contour(nipple_mask, colors='darkred', linewidths=3, alpha=0.9)
    if areola_mask is not None: axes[0].contour(areola_mask, colors='coral', linewidths=2, alpha=0.7)
    plt.colorbar(im1, ax=axes[0], label='ε').set_ticks(np.linspace(vmin_eps, vmax_eps, 6))
    anatomy = np.zeros(breast_mask.shape)
    anatomy[breast_mask] = 1; anatomy[areola_mask] = 2; anatomy[nipple_mask] = 3; anatomy[body_mask] = 4
    im2 = axes[1].imshow(anatomy, cmap='tab10', vmin=0, vmax=4)
    axes[1].set_title('Анатомическая структура', fontsize=12, fontweight='bold')
    axes[1].text(0.02, 0.95, '1 - Жировая ткань\n2 - Ареола\n3 - Сосок\n4 - Тело', transform=axes[1].transAxes, fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.colorbar(im2, ax=axes[1], label='Тип ткани')
    plt.tight_layout()
    plt.savefig('08_breast_anatomy.png', dpi=150, bbox_inches='tight')
    plt.close()  

def plot_temperature_gradient(temp_map, breast_mask, tumor_center=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    h, w = temp_map.shape; center_x = w // 2
    temp_profile = temp_map[:, center_x].copy(); temp_profile[~breast_mask[:, center_x]] = np.nan
    y_coords = np.arange(h); valid_mask = breast_mask[:, center_x]
    valid_profile = temp_profile[valid_mask]
    vmin_prof, vmax_prof = (np.nanmin(valid_profile), np.nanmax(valid_profile)) if len(valid_profile) > 0 else (33, 39)
    axes[0].plot(valid_profile, y_coords[valid_mask], 'b-', linewidth=2.5)
    axes[0].fill_betweenx(y_coords[valid_mask], valid_profile, vmin_prof, alpha=0.3)
    axes[0].invert_yaxis(); axes[0].set_xlabel('Температура (°C)', fontsize=11); axes[0].set_ylabel('Глубина (пиксели)', fontsize=11)
    axes[0].set_title(f'Температурный градиент\n[{vmin_prof:.1f}°C — {vmax_prof:.1f}°C]', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3); axes[0].axvline(34.5, color='green', linestyle='--', label='Поверхность (~34.5°C)')
    axes[0].axvline(37.0, color='red', linestyle='--', label='Грудная стенка (~37°C)'); axes[0].legend(loc='lower right')
    valid_temps = temp_map[breast_mask]
    axes[1].hist(valid_temps, bins=40, color='steelblue', edgecolor='black', alpha=0.7)
    axes[1].axvline(valid_temps.mean(), color='red', linestyle='--', linewidth=2, label=f'Среднее: {valid_temps.mean():.2f}°C')
    axes[1].axvline(valid_temps.min(), color='green', linestyle=':', linewidth=2, label=f'Мин: {valid_temps.min():.2f}°C')
    axes[1].axvline(valid_temps.max(), color='orange', linestyle=':', linewidth=2, label=f'Макс: {valid_temps.max():.2f}°C')
    axes[1].set_xlabel('Температура (°C)', fontsize=11); axes[1].set_ylabel('Количество пикселей', fontsize=11)
    axes[1].set_title(f'Распределение температур\n[{valid_temps.min():.1f}°C — {valid_temps.max():.1f}°C]', fontsize=12, fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('09_temperature_gradient.png', dpi=150, bbox_inches='tight')
    plt.close()  

def plot_temperature_contours(temp_map, breast_mask, tumor_center=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    temp_display = temp_map.copy(); temp_display[~breast_mask] = np.nan
    vmin_temp, vmax_temp = np.nanmin(temp_display), np.nanmax(temp_display)
    im = ax.imshow(temp_display, cmap='jet', vmin=vmin_temp, vmax=vmax_temp, interpolation='gaussian')
    contour_levels = np.arange(np.ceil(vmin_temp*10)/10, np.floor(vmax_temp*10)/10 + 0.1, 0.3)
    if len(contour_levels) > 1:
        cs = ax.contour(temp_display, levels=contour_levels, colors='white', linewidths=1.0, alpha=0.8)
        ax.clabel(cs, inline=True, fontsize=7, fmt='%.1f°C')
    if tumor_center: ax.plot(tumor_center[1], tumor_center[0], 'r+', markersize=15, markeredgewidth=2, label='Опухоль'); ax.legend(loc='lower right')
    ax.set_title(f'Изотермы температуры\n[{vmin_temp:.1f}°C — {vmax_temp:.1f}°C]', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Температура (°C)').set_ticks(np.linspace(vmin_temp, vmax_temp, 6))
    plt.tight_layout()
    plt.savefig('10_temperature_contours.png', dpi=150, bbox_inches='tight')
    plt.close()  

def plot_temperature_difference_map(temp_map, tissue_type_map, breast_mask):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    tissue_temps = {i: np.mean(temp_map[tissue_type_map == i]) for i in range(1, 8) if np.sum(tissue_type_map == i) > 100}
    avg_temp = np.mean(temp_map[breast_mask])
    temp_diff = temp_map - avg_temp; temp_diff[~breast_mask] = np.nan
    vmax_diff = max(abs(np.nanmin(temp_diff)), abs(np.nanmax(temp_diff)))
    im1 = axes[0].imshow(temp_diff, cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff, interpolation='gaussian')
    axes[0].set_title(f'Отклонение от средней\n[{np.nanmin(temp_diff):.2f}°C — {np.nanmax(temp_diff):.2f}°C]', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='ΔT (°C)').set_ticks(np.linspace(-vmax_diff, vmax_diff, 5))
    tissue_names = {1: 'Подкожный\nжир', 2: 'Железистая', 3: 'Внутрижелез.\nжир', 4: 'Ретромаммарный', 5: 'Соединительная', 6: 'Протоки', 7: 'Дольки'}
    temps = [tissue_temps.get(i, 0) for i in range(1, 8)]; names = [tissue_names.get(i, '') for i in range(1, 8)]
    axes[1].bar(range(len(temps)), temps, color=['#F9F0D9', '#D4A5A5', '#F4E4C1', '#E8D5A3', '#E0C0C0', '#C48585', '#B87070'])
    axes[1].set_xticks(range(len(names))); axes[1].set_xticklabels(names, fontsize=8)
    axes[1].axhline(avg_temp, color='red', linestyle='--', linewidth=2, label=f'Средняя: {avg_temp:.2f}°C')
    axes[1].set_ylabel('Температура (°C)', fontsize=11); axes[1].set_title('Температура по типам тканей', fontsize=12, fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3, axis='y'); axes[1].set_ylim(avg_temp - 2, avg_temp + 2)
    plt.tight_layout()
    plt.savefig('11_temperature_difference.png', dpi=150, bbox_inches='tight')
    plt.close()  

def plot_sensitivity_kernels(model, breast_mask, scan_positions, n_show=5):
    fig, axes = plt.subplots(1, n_show, figsize=(20, 4))
    if n_show == 1: axes = [axes]
    step = max(1, len(scan_positions) // n_show)
    for i, ax in enumerate(axes):
        idx = min(i * step, len(scan_positions) - 1)
        pos = scan_positions[idx]; kernel = model.compute_sensitivity_kernel(breast_mask, pos)
        im = ax.imshow(kernel, cmap='viridis', vmin=0, vmax=np.max(kernel)*1.2)
        ax.plot(pos[1], pos[0], 'r*', markersize=15, label='Антенна')
        ax.set_title(f'Антенна #{idx+1}\nПозиция: ({pos[0]}, {pos[1]})', fontsize=10)
        ax.legend(loc='upper right', fontsize=8); plt.colorbar(im, ax=ax, label='Норм. чувствительность')
    plt.suptitle('Функции чувствительности антенн', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('02_sensitivity_kernels.png', dpi=150, bbox_inches='tight')
    plt.close()  

def plot_measurement_data(Tb_data, Tb_noisy, emissivity_avg, scan_positions):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x_antenna = [pos[1] for pos in scan_positions]
    axes[0].plot(x_antenna, Tb_data, 'bo-', linewidth=2, markersize=8, label='Без шума')
    axes[0].plot(x_antenna, Tb_noisy, 'rs--', linewidth=2, markersize=8, label='С шумом')
    axes[0].set_xlabel('Позиция антенны (X)', fontsize=11); axes[0].set_ylabel('Яркостная температура Tb (K)', fontsize=11)
    axes[0].set_title('Измерения яркостной температуры', fontsize=12, fontweight='bold'); axes[0].legend(loc='best'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(x_antenna, emissivity_avg, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Позиция антенны (X)', fontsize=11); axes[1].set_ylabel('Средний коэффициент излучения', fontsize=11)
    axes[1].set_title('Коэффициент излучения по антеннам', fontsize=12, fontweight='bold'); axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=emissivity_avg.mean(), color='r', linestyle='--', label=f'Среднее: {emissivity_avg.mean():.3f}'); axes[1].legend(loc='best')
    plt.tight_layout()
    plt.savefig('03_measurement_data.png', dpi=150, bbox_inches='tight')
    plt.close()  

def plot_temperature_histogram(temp_true, temp_recon, breast_mask):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    true_vals = temp_true[breast_mask]
    recon_vals = temp_recon[breast_mask]
    
    hist_range1 = (34.0, 39.0)  
    hist_range2 = (35.0, 38.0)  
    # Гистограмма истинных температур
    axes[0].hist(true_vals, bins=30, range=hist_range1, 
                 color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(true_vals.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Среднее: {true_vals.mean():.2f}°C')
    axes[0].set_xlabel('Температура (°C)', fontsize=11)
    axes[0].set_ylabel('Количество пикселей', fontsize=11)
    axes[0].set_title(f'Распределение истинных температур', fontsize=12, fontweight='bold')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    # Гистограмма реконструированных температур
    axes[1].hist(recon_vals, bins=30, range=hist_range2, 
                 color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(recon_vals.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Среднее: {recon_vals.mean():.2f}°C')
    axes[1].set_xlabel('Температура (°C)', fontsize=11)
    axes[1].set_ylabel('Количество пикселей', fontsize=11)
    axes[1].set_title(f'Распределение реконструированных температур', fontsize=12, fontweight='bold')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('04_temperature_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()

def plot_cross_section(temp_true, temp_recon, breast_mask, tumor_center=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if tumor_center:
        h, w = temp_true.shape
        for ax, axis in zip(axes, ['x', 'y']):
            if axis == 'x':
                slice_idx = int(tumor_center[0]); vals_true, vals_recon = temp_true[slice_idx, :], temp_recon[slice_idx, :]
                mask_slice = breast_mask[slice_idx, :]; coords = np.arange(w); title = f'Горизонтальный срез Y={slice_idx}'
            else:
                slice_idx = int(tumor_center[1]); vals_true, vals_recon = temp_true[:, slice_idx], temp_recon[:, slice_idx]
                mask_slice = breast_mask[:, slice_idx]; coords = np.arange(h); title = f'Вертикальный срез X={slice_idx}'
            vmin_s, vmax_s = np.nanmin(np.concatenate([vals_true[mask_slice], vals_recon[mask_slice]])), np.nanmax(np.concatenate([vals_true[mask_slice], vals_recon[mask_slice]]))
            ax.plot(coords[mask_slice], vals_true[mask_slice], 'b-', linewidth=2, label='Истинная T')
            ax.plot(coords[mask_slice], vals_recon[mask_slice], 'r--', linewidth=2, label='Реконструированная T')
            ax.axvline(tumor_center[1] if axis == 'x' else tumor_center[0], color='green', linestyle=':', linewidth=2, label='Центр опухоли')
            ax.set_xlabel('Позиция (пиксели)', fontsize=11); ax.set_ylabel('Температура (°C)', fontsize=11)
            ax.set_title(f'{title}\n[{vmin_s:.1f}°C — {vmax_s:.1f}°C]', fontsize=12, fontweight='bold')
            ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(vmin_s - 0.5, vmax_s + 0.5)
    else:
        for ax in axes: ax.axis('off'); ax.text(0.5, 0.5, 'Опухоль не найдена', ha='center', va='center', fontsize=14, transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig('05_cross_section.png', dpi=150, bbox_inches='tight')
    plt.close()  

def plot_emissivity_map(eps_map, breast_mask):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sqrt_eps = np.sqrt(np.maximum(eps_map, 1.0)); gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0); emissivity = 1.0 - gamma**2
    eps_inside = eps_map[breast_mask]; vmin_eps, vmax_eps = np.min(eps_inside), np.max(eps_inside)
    im1 = axes[0].imshow(eps_map, cmap='viridis', vmin=vmin_eps, vmax=vmax_eps, interpolation='gaussian')
    axes[0].set_title(f'Диэлектрическая проницаемость\n[{vmin_eps:.1f} — {vmax_eps:.1f}]', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[0], label='ε').set_ticks(np.linspace(vmin_eps, vmax_eps, 6))
    emissivity_display = emissivity.copy(); emissivity_display[~breast_mask] = np.nan
    im2 = axes[1].imshow(emissivity_display, cmap='plasma', vmin=0.5, vmax=1.0, interpolation='gaussian')
    axes[1].set_title('Коэффициент излучения (Emissivity)\n[0.5 — 1.0]', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Emissivity').set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    plt.tight_layout()
    plt.savefig('06_emissivity_map.png', dpi=150, bbox_inches='tight')
    plt.close()  

def plot_antenna_coverage(breast_mask, scan_positions, model, temp_recon=None, filename='15_antenna_coverage.png'):
    """
    Визуализация покрытия антенн: позиции + направление + зона чувствительности.
    
    Параметры:
    - breast_mask: бинарная маска груди
    - scan_positions: список кортежей (y, x) с позициями антенн
    - model: объект BreastRadiometryModelReal (для вычисления ядер)
    - temp_recon: опционально, реконструированная температура для фона
    - filename: имя сохраняемого файла
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Фон: либо маска, либо реконструированная температура
    if temp_recon is not None:
        display = temp_recon.copy()
        display[~breast_mask] = np.nan
        im = ax.imshow(display, cmap='jet', vmin=33.0, vmax=40.0, alpha=0.7, interpolation='gaussian')
        plt.colorbar(im, ax=ax, label='T (°C)', fraction=0.046, pad=0.04)
    else:
        ax.imshow(breast_mask, cmap='gray', alpha=0.3)
    
    # Контур груди
    from scipy.ndimage import binary_erosion
    boundary = binary_erosion(breast_mask, iterations=2) ^ breast_mask
    ax.contour(boundary, colors='black', linewidths=1.5, alpha=0.8)
    
    # Отрисовка антенн и их ядер чувствительности
    for i, (y, x) in enumerate(scan_positions):
        # Маркер позиции антенны
        ax.plot(x, y, 'r*', markersize=12, markeredgewidth=2, label='Антенна' if i==0 else "")
        
        # Вычисляем ядро чувствительности для этой антенны
        kernel = model.compute_sensitivity_kernel(breast_mask, (y, x))
        
        # Нормируем для визуализации (берем верхние 30% значений)
        kernel_vis = kernel.copy()
        threshold = np.percentile(kernel_vis[kernel_vis > 0], 70)
        kernel_vis[kernel_vis < threshold] = 0
        
        # Отображаем зону чувствительности как полупрозрачный контур
        if np.any(kernel_vis > 0):
            # Создаем контуры для ядра
            levels = np.linspace(threshold, kernel_vis.max(), 3)
            ax.contour(kernel_vis, levels=levels, colors='red', linewidths=0.8, alpha=0.6,
                      extent=[0, kernel_vis.shape[1], kernel_vis.shape[0], 0])
        
        # Стрелка направления (антенна "смотрит" вглубь ткани, т.е. вниз по оси Y)
        arrow_len = 15  # длина стрелки в пикселях
        ax.arrow(x, y, 0, arrow_len, head_width=3, head_length=5, 
                fc='red', ec='red', alpha=0.7, linewidth=1.5)
    
    # Настройки графика
    ax.set_xlabel('X (пиксели)', fontsize=11)
    ax.set_ylabel('Y (пиксели)', fontsize=11)
    ax.set_title(f'Покрытие антеннами ({len(scan_positions)} шт.)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"График покрытия антенн сохранён: {filename}")