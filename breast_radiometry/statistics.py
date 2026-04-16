import numpy as np

def print_full_statistics(temp_true, temp_recon, breast_mask, Tb_data, Tb_noisy, emissivity_avg, eps_map, cond_map, model, tissue_type_map=None):

    valid_true, valid_recon = temp_true[breast_mask], temp_recon[breast_mask]
    print(f"\nДиэлектрические св-ва (внутри груди):\n   EPS:  {np.mean(eps_map[breast_mask]):6.2f} ± {np.std(eps_map[breast_mask]):.2f}\n   COND: {np.mean(cond_map[breast_mask]):6.2f} ± {np.std(cond_map[breast_mask]):.2f} См/м")
    print(f"\nСтатистика температуры:\n   Истинная T:        {valid_true.mean():6.2f} ± {valid_true.std():.2f} °C\n   Реконструированная: {valid_recon.mean():6.2f} ± {valid_recon.std():.2f} °C\n   Смещение (bias):    {valid_recon.mean() - valid_true.mean():+.2f} °C\n   Мин T:              {valid_true.min():6.2f} °C\n   Макс T:             {valid_true.max():6.2f} °C\n   Диапазон:           {valid_true.max() - valid_true.min():.2f} °C")
    
    temp_masked = temp_true.copy(); temp_masked[~breast_mask] = 0
    grad_y, grad_x = np.gradient(temp_masked)
    grad_inside = np.sqrt(grad_y**2 + grad_x**2)[breast_mask]
    #print(f"   Плавность (средн. градиент): {grad_inside.mean():.4f} °C/пиксель\n   Плавность (макс. градиент):  {grad_inside.max():.4f} °C/пиксель")
    
    abs_error = np.abs(valid_true - valid_recon)
    print(f"\nОшибки реконструкции:\n   Средняя (MAE):      {abs_error.mean():.2f} °C\n   Максимальная:       {abs_error.max():.2f} °C\n   RMSE:               {np.sqrt(np.mean(abs_error**2)):.2f} °C\n   Медианная:          {np.median(abs_error):.2f} °C")
    
    print(f"\nИзмерения радиотермометра:\n   Количество антенн:  {len(Tb_data)}\n   Tb (мин):           {Tb_noisy.min():.2f} K\n   Tb (макс):          {Tb_noisy.max():.2f} K\n   Tb (среднее):       {Tb_noisy.mean():.2f} K\n   Emissivity (средн.): {emissivity_avg.mean():.3f} ± {emissivity_avg.std():.3f}")
    
    print("\nОпухоль:")
    if hasattr(model, 'tumor_center') and model.tumor_center:
        ty, tx = model.tumor_center
        y_coords, x_coords = np.ogrid[:temp_true.shape[0], :temp_true.shape[1]]
        tumor_region = (x_coords - tx)**2 + (y_coords - ty)**2 <= 10**2
        if np.sum(tumor_region & breast_mask) > 0:
            tumor_true = temp_true[tumor_region & breast_mask]
            tumor_recon = temp_recon[tumor_region & breast_mask]
            print(f"   Координаты:         Y={ty}, X={tx}\n   T в опухоли (истина): {tumor_true.mean():.2f} °C\n   T в опухоли (рекон):  {tumor_recon.mean():.2f} °C\n   Контраст опухоли:     {tumor_true.mean() - valid_true.mean():.2f} °C")
        else: print("   Опухоль за пределами груди")
    else: print("   Опухоль не была создана")
    
    if tissue_type_map is not None:
        print("\nТемпература в зависимости от типа ткани:")
        tissue_names = {1: 'Подкожный жир', 2: 'Железистая', 3: 'Внутрижелез. жир', 4: 'Ретромаммарный', 5: 'Соединительная', 6: 'Протоки', 7: 'Дольки'}
        for i, name in tissue_names.items():
            mask = tissue_type_map == i
            if np.sum(mask) > 100:
                t_mean, t_std = np.mean(temp_true[mask]), np.std(temp_true[mask])
                pct = np.sum(mask) / np.sum(breast_mask) * 100
                print(f"   {name:15s}: {t_mean:5.2f} ± {t_std:.2f} °C ({pct:5.1f}%)")
