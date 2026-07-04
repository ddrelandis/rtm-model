"""
Анатомически точная 3D модель радиотермометрии головного мозга
Содержит детализированную анатомию, включая извилины, борозды, череп и т.д.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation, distance_transform_edt, zoom
import time
import math
from scipy.spatial import ConvexHull

class BrainRadiometryModel3D:
    """
    3D модель радиотермометрии головного мозга с максимальной анатомической точностью.
    
    Слои (от внешнего к внутреннему):
    1. Кожа головы (scalp) - 2-3 мм
    2. Подкожный жир (subcutaneous fat) - 3-5 мм
    3. Мышцы (muscle) - 2-4 мм
    4. Череп (skull) - 6-7 мм (компактная кость + губчатая кость)
    5. Твёрдая мозговая оболочка (dura mater) - 0.5-1.0 мм
    6. Субарахноидальное пространство (CSF) - 2-3 мм
    7. Серое вещество (grey matter) - 2-4 мм
    8. Белое вещество (white matter) - основная масса
    9. Желудочковая система (ventricles) - ликвор
    10. Мозжечок (cerebellum)
    11. Ствол мозга (brainstem)
    """
    
    def __init__(self, freq_ghz=2.0, resolution_mm=1.5, use_real_anatomy=False):
        """
        freq_ghz: рабочая частота (1-3 ГГц типично для мозга)
        resolution_mm: разрешение сетки в мм
        use_real_anatomy: использовать реальные анатомические данные (замедляет вычисления)
        """
        self.freq = freq_ghz * 1e9
        self.c = 3e8
        self.res = resolution_mm / 1000.0
        self.use_real_anatomy = use_real_anatomy
        self.tumor_center = None
        
        # Диэлектрические свойства тканей мозга на 2 ГГц (Gabriel et al.)
        self.tissue_props = {
            'scalp': {
                'mean_eps': 38.0, 'std_eps': 4.0,
                'mean_cond': 1.0, 'std_cond': 0.2,
                'temp_base': 33.5, 'name': 'Скальп (кожа головы)'
            },
            'fat': {
                'mean_eps': 5.5, 'std_eps': 0.5,
                'mean_cond': 0.10, 'std_cond': 0.03,
                'temp_base': 34.0, 'name': 'Подкожный жир'
            },
            'muscle': {
                'mean_eps': 45.0, 'std_eps': 5.0,
                'mean_cond': 1.8, 'std_cond': 0.3,
                'temp_base': 35.5, 'name': 'Мышцы'
            },
            'skull_compact': {
                'mean_eps': 12.0, 'std_eps': 2.0,
                'mean_cond': 0.03, 'std_cond': 0.01,
                'temp_base': 35.0, 'name': 'Компактная кость черепа'
            },
            'skull_spongy': {
                'mean_eps': 15.0, 'std_eps': 2.5,
                'mean_cond': 0.05, 'std_cond': 0.02,
                'temp_base': 35.5, 'name': 'Губчатая кость черепа'
            },
            'dura_mater': {
                'mean_eps': 40.0, 'std_eps': 4.0,
                'mean_cond': 1.2, 'std_cond': 0.15,
                'temp_base': 36.0, 'name': 'Твёрдая мозговая оболочка'
            },
            'csf': {
                'mean_eps': 73.0, 'std_eps': 3.0,
                'mean_cond': 2.2, 'std_cond': 0.15,
                'temp_base': 37.2, 'name': 'Ликвор (CSF)'
            },
            'gray_matter': {
                'mean_eps': 48.0, 'std_eps': 5.0,
                'mean_cond': 1.8, 'std_cond': 0.25,
                'temp_base': 37.0, 'name': 'Серое вещество (кора)'
            },
            'white_matter': {
                'mean_eps': 38.0, 'std_eps': 4.0,
                'mean_cond': 1.2, 'std_cond': 0.20,
                'temp_base': 37.0, 'name': 'Белое вещество'
            },
            'cerebellum': {
                'mean_eps': 48.0, 'std_eps': 5.0,
                'mean_cond': 1.8, 'std_cond': 0.25,
                'temp_base': 37.0, 'name': 'Мозжечок'
            },
            'brainstem': {
                'mean_eps': 45.0, 'std_eps': 5.0,
                'mean_cond': 1.6, 'std_cond': 0.20,
                'temp_base': 37.0, 'name': 'Ствол мозга'
            },
            'tumor': {
                'mean_eps': 55.0, 'std_eps': 7.0,
                'mean_cond': 3.0, 'std_cond': 0.5,
                'temp_base': 38.5, 'name': 'Опухоль'
            },
            'air': {
                'mean_eps': 1.0, 'std_eps': 0.0,
                'mean_cond': 0.0, 'std_cond': 0.0,
                'temp_base': 20.0, 'name': 'Воздух'
            }
        }
    
    def create_realistic_head_shape(self, shape, scale=1.0):
        """
        Создает реалистичную форму головы с двумя полушариями мозга.
        """
        d, h, w = shape
        z, y, x = np.ogrid[:d, :h, :w]
        center_z, center_y, center_x = d * 0.5, h * 0.5, w * 0.5
        
        # Внешняя форма головы (череп + скальп)
        head_rz = d * 0.47 * scale
        head_ry = h * 0.47 * scale
        head_rx = w * 0.47 * scale
        
        z_norm = (z - center_z) / head_rz
        y_norm = (y - center_y) / head_ry
        x_norm = (x - center_x) / head_rx
        
        # Голова как эллипсоид
        head_mask = (z_norm**2 + y_norm**2 + x_norm**2) <= 1.0
        
        return head_mask
    
    def create_brain_hemispheres(self, head_mask, shape):
        """
        Процедурная генерация реалистичной формы мозга.
        Использует сумму гауссиан (metaballs) для формирования долей и продольной борозды.
        """
        d, h, w = shape
        z, y, x = np.ogrid[:d, :h, :w]
        cz, cy, cx = d * 0.48, h * 0.50, w * 0.50
        
        # Функция-гауссиана для "лепки" долей
        def gauss(z0, y0, x0, sz, sy, sx, amp=1.0):
            return amp * np.exp(-((z-z0)**2/(2*sz**2) + (y-y0)**2/(2*sy**2) + (x-x0)**2/(2*sx**2)))
        
        # === ЛЕВОЕ ПОЛУШАРИЕ ===
        # Лобная доля (Frontal)
        l_frontal = gauss(cz - d*0.18, cy - h*0.12, cx - w*0.12, d*0.14, h*0.13, w*0.11, 1.0)
        # Теменная доля (Parietal)
        l_parietal = gauss(cz + d*0.05, cy - h*0.14, cx - w*0.13, d*0.12, h*0.12, w*0.12, 0.95)
        # Височная доля (Temporal)
        l_temporal = gauss(cz + d*0.02, cy + h*0.16, cx - w*0.14, d*0.10, h*0.09, w*0.10, 0.85)
        # Затылочная доля (Occipital)
        l_occipital = gauss(cz + d*0.20, cy - h*0.02, cx - w*0.10, d*0.10, h*0.10, w*0.09, 0.90)
        
        # === ПРАВОЕ ПОЛУШАРИЕ (зеркально) ===
        r_frontal = gauss(cz - d*0.18, cy + h*0.12, cx + w*0.12, d*0.14, h*0.13, w*0.11, 1.0)
        r_parietal = gauss(cz + d*0.05, cy + h*0.14, cx + w*0.13, d*0.12, h*0.12, w*0.12, 0.95)
        r_temporal = gauss(cz + d*0.02, cy - h*0.16, cx + w*0.14, d*0.10, h*0.09, w*0.10, 0.85)
        r_occipital = gauss(cz + d*0.20, cy + h*0.02, cx + w*0.10, d*0.10, h*0.10, w*0.09, 0.90)
        
        # Суммируем все доли
        brain_field = (l_frontal + l_parietal + l_temporal + l_occipital +
                       r_frontal + r_parietal + r_temporal + r_occipital)
        
        # Пороговое значение для формирования поверхности
        threshold = 0.35
        brain_mask = brain_field > threshold
        
        # Ограничиваем головой
        brain_mask &= head_mask
        
        # === ПРОДОЛЬНАЯ БОРОЗДА (Longitudinal Fissure) ===
        # Вырезаем глубокую щель между полушариями
        fissure_width = max(2, int(2.5 * (h / 140.0)))
        fissure_depth_factor = np.clip((brain_field - threshold) / 0.3, 0, 1)
        fissure_mask = (np.abs(x - cx) < fissure_width) & (fissure_depth_factor > 0.2)
        brain_mask &= ~fissure_mask
        
        # === ШУМ ДЛЯ ИЗВИЛИН (Gyri/Sulci) ===
        # Высокочастотный шум для деформации поверхности коры
        np.random.seed(123)
        noise = np.random.randn(d, h, w).astype(np.float32)
        noise = gaussian_filter(noise, sigma=max(2.0, 3.0 * (h / 140.0)))
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        
        # Деформируем границу: там где шум высокий — ткань выступает (извилина),
        # где низкий — углубление (борозда)
        deformation = (noise - 0.5) * 0.15
        deformed_field = brain_field + deformation
        
        # Пересоздаём маску с учётом деформации
        cortex_surface = deformed_field > threshold
        cortex_surface &= head_mask
        cortex_surface &= ~fissure_mask
        
        # Белое вещество — внутренняя часть (эрозия коры)
        cortex_thickness = max(2, int(3.5 * (h / 140.0)))
        white_matter = binary_erosion(cortex_surface, iterations=cortex_thickness)
        cortex_mask = cortex_surface & ~white_matter
        
        return cortex_surface, fissure_mask, noise, white_matter

    def create_cortical_folding(self, brain_mask, shape):
        """
        Создает реалистичные извилины и борозды коры.
        """
        d, h, w = shape
        z, y, x = np.ogrid[:d, :h, :w]
        center_z, center_y, center_x = d * 0.5, h * 0.5, w * 0.5
        
        # Создаём шум для моделирования извилин
        np.random.seed(42)
        noise_field = np.random.randn(d, h, w)
        noise_field = gaussian_filter(noise_field, sigma=6.0)
        noise_field = (noise_field - noise_field.min()) / (noise_field.max() - noise_field.min())
        
        # ===== ОСНОВНЫЕ БОРОЗДЫ (Major Sulci) =====
        
        # Центральная борозда (Rolandic fissure) - разделяет лобную и теменную доли
        central_sulcus_left = ((np.abs(z - center_z) < max(3, int(5*(h/140.0)))) & 
                               (x < center_x - 5) &
                               (y > center_y - h*0.3) & (y < center_y + h*0.1))
        
        central_sulcus_right = ((np.abs(z - center_z) < max(3, int(5*(h/140.0)))) & 
                                (x > center_x + 5) &
                                (y > center_y - h*0.3) & (y < center_y + h*0.1))
        
        # Латеральная борозда (Sylvian fissure) - разделяет височную и лобную/теменную доли
        sylvian_fissure_left = ((y > center_y + h*0.05) & (y < center_y + h*0.15) &
                                (x < center_x - 5) &
                                (z > center_z - h*0.3) & (z < center_z + h*0.3))
        
        sylvian_fissure_right = ((y > center_y + h*0.05) & (y < center_y + h*0.15) &
                                 (x > center_x + 5) &
                                 (z > center_z - h*0.3) & (z < center_z + h*0.3))
        
        # Теменно-затылочная борозда
        parieto_occipital_left = ((z > center_z + h*0.15) & 
                                  (x < center_x - 5) &
                                  (y > center_y - h*0.2) & (y < center_y + h*0.2))
        
        parieto_occipital_right = ((z > center_z + h*0.15) & 
                                   (x > center_x + 5) &
                                   (y > center_y - h*0.2) & (y < center_y + h*0.2))
        
        # Объединяем все борозды
        major_sulci = (central_sulcus_left | central_sulcus_right |
                      sylvian_fissure_left | sylvian_fissure_right |
                      parieto_occipital_left | parieto_occipital_right)
        
        # ===== МАЛЫЕ БОРОЗДЫ (Minor Sulci) из шума =====
        minor_sulci = noise_field < 0.35
        minor_sulci &= brain_mask
        
        # ===== ИЗВИЛИНЫ (Gyri) =====
        gyri = noise_field > 0.65
        gyri &= brain_mask
        
        # Создаем кору с бороздами
        cortex_with_sulci = brain_mask & ~major_sulci & ~minor_sulci
        
        # Толщина коры
        cortex_thickness = max(2, int(3 * (h/140.0)))
        cortex_mask = binary_erosion(cortex_with_sulci, iterations=cortex_thickness) ^ cortex_with_sulci
        
        # Белое вещество - всё что внутри коры
        white_matter = binary_erosion(cortex_with_sulci, iterations=cortex_thickness)
        
        return cortex_mask, major_sulci | minor_sulci, noise_field, white_matter
    
    def create_ventricular_system(self, shape):
        """
        Создает желудочковую систему мозга с реалистичными размерами и формой.
        """
        d, h, w = shape
        z, y, x = np.ogrid[:d, :h, :w]
        center_z, center_y, center_x = d//2, h//2, w//2
        
        # Боковые желудочки
        ventricle_rz = d * 0.15
        ventricle_ry = h * 0.12
        ventricle_rx = w * 0.08
        ventricle_offset_y = h * 0.08
        ventricle_offset_x = w * 0.08
        
        # Левый боковой желудочек
        left_ventricle = ((z - center_z)**2 / ventricle_rz**2 + 
                         (y - (center_y - ventricle_offset_y))**2 / ventricle_ry**2 + 
                         (x - (center_x - ventricle_offset_x))**2 / ventricle_rx**2) <= 1.0
        
        # Правый боковой желудочек
        right_ventricle = ((z - center_z)**2 / ventricle_rz**2 + 
                          (y - (center_y + ventricle_offset_y))**2 / ventricle_ry**2 + 
                          (x - (center_x + ventricle_offset_x))**2 / ventricle_rx**2) <= 1.0
        
        # Третий желудочек (маленький)
        third_ventricle = ((z - center_z)**2 / (ventricle_rz*0.5)**2 + 
                          (y - center_y)**2 / (ventricle_ry*0.3)**2 + 
                          (x - center_x)**2 / (ventricle_rx*0.5)**2) <= 1.0
        
        # Четвёртый желудочек
        fourth_ventricle = ((z - center_z*0.8)**2 / (ventricle_rz*0.4)**2 + 
                           (y - center_y)**2 / (ventricle_ry*0.3)**2 + 
                           (x - center_x)**2 / (ventricle_rx*0.4)**2) <= 1.0
        
        # Водопровод мозга
        cerebral_aqueduct = ((z - center_z*0.85)**2 / (ventricle_rz*0.2)**2 + 
                             (y - center_y)**2 / (ventricle_ry*0.1)**2 + 
                             (x - center_x)**2 / (ventricle_rx*0.2)**2) <= 1.0
        
        ventricles = (left_ventricle | right_ventricle | 
                      third_ventricle | fourth_ventricle | 
                      cerebral_aqueduct)
        
        return ventricles
    
    def create_cerebellum(self, shape, brain_mask):
        """
        Создает реалистичную форму мозжечка с характерными дольками.
        """
        d, h, w = shape
        z, y, x = np.ogrid[:d, :h, :w]
        center_z, center_y, center_x = d * 0.5, h * 0.5, w * 0.5
        
        # Мозжечок расположен сзади и снизу
        cerebellum_center_z = int(d * 0.75)
        cerebellum_center_y = int(h * 0.5)
        cerebellum_center_x = int(w * 0.5)
        
        # Мозжечок шире, чем выше
        cerebellum_rz = int(d * 0.10)
        cerebellum_ry = int(h * 0.12)
        cerebellum_rx = int(w * 0.20)
        
        # Базовая форма мозжечка (сплюснутый эллипсоид)
        cerebellum_mask = ((z - cerebellum_center_z)**2 / cerebellum_rz**2 + 
                          (y - cerebellum_center_y)**2 / cerebellum_ry**2 + 
                          (x - cerebellum_center_x)**2 / cerebellum_rx**2) <= 1.0
        
        # ===== ДОЛЬКИ МОЗЖЕЧКА (Folia) =====
        # Характерные горизонтальные складки
        folia_spacing = max(3, int(4 * (h/140.0)))
        folia_mask = np.zeros(shape, dtype=bool)
        
        for i in range(-5, 6):
            folia_y = cerebellum_center_y + i * folia_spacing
            folia_strip = (np.abs(y - folia_y) < folia_spacing * 0.4)
            folia_mask |= folia_strip
        
        cerebellum_mask &= folia_mask
        
        # Разделяем на два полушария мозжечка
        cerebellum_fissure = np.abs(x - center_x) < max(1, int(2 * (h/140.0)))
        cerebellum_mask &= ~cerebellum_fissure
        
        return cerebellum_mask
    
    def create_brainstem(self, shape, brain_mask):
        """
        Создает реалистичную форму ствола мозга.
        """
        d, h, w = shape
        z, y, x = np.ogrid[:d, :h, :w]
        
        # Центр ствола мозга
        brainstem_center_z = int(d * 0.90)
        brainstem_center_y = int(h * 0.5)
        brainstem_center_x = int(w * 0.5)
        
        # Размеры ствола
        brainstem_rz = int(d * 0.07)
        brainstem_ry = int(h * 0.05)
        brainstem_rx = int(w * 0.05)
        
        # Базовая форма ствола
        brainstem_mask = ((z - brainstem_center_z)**2 / brainstem_rz**2 + 
                         (y - brainstem_center_y)**2 / brainstem_ry**2 + 
                         (x - brainstem_center_x)**2 / brainstem_rx**2) <= 1.0
        
        # Мост (pons)
        pons_center_z = int(d * 0.80)
        pons_mask = ((z - pons_center_z)**2 / (brainstem_rz*0.7)**2 + 
                    (y - brainstem_center_y)**2 / (brainstem_ry*0.8)**2 + 
                    (x - brainstem_center_x)**2 / (brainstem_rx*0.7)**2) <= 1.0
        
        # Добавляем мост к стволу
        brainstem_mask |= pons_mask
        
        # Привязываем к мозгу
        brainstem_mask &= brain_mask
        
        return brainstem_mask
    
    def create_anatomical_brain_phantom(self, shape=(100, 140, 140), tumor_radius=8, tumor_pos=None):
        """
        Создает анатомически точный 3D фантом головы с мозгом.
        """
        start_time = time.time()
        d, h, w = shape
        print(f"🔍 Генерация анатомического фантома мозга: {d}x{h}x{w} вокселей")
        z, y, x = np.ogrid[:d, :h, :w]
        # 1. Форма головы
        head_mask = self.create_realistic_head_shape(shape, scale=1.0)
        
        # 2. Череп
        skull_thickness = max(4, int(7 * (h/140.0)))
        skull_compact_outer = binary_erosion(head_mask, iterations=skull_thickness) ^ head_mask
        
        skull_spongy_thickness = max(2, int(3 * (h/140.0)))
        skull_spongy = binary_erosion(binary_erosion(head_mask, iterations=skull_thickness), 
                                     iterations=skull_spongy_thickness) ^ \
                       binary_erosion(head_mask, iterations=skull_thickness)
        
        skull_compact_inner_thickness = max(2, int(2 * (h/140.0)))
        skull_compact_inner = binary_erosion(binary_erosion(head_mask, iterations=skull_thickness + skull_spongy_thickness), 
                                             iterations=skull_compact_inner_thickness) ^ \
                              binary_erosion(head_mask, iterations=skull_thickness + skull_spongy_thickness)
        
        brain_outer = binary_erosion(head_mask, iterations=skull_thickness + skull_spongy_thickness + skull_compact_inner_thickness)
        
        # 3. Твёрдая мозговая оболочка
        dura_thickness = max(1, int(1 * (h/140.0)))
        dura_mask = binary_erosion(brain_outer, iterations=dura_thickness) ^ brain_outer
        brain_inner = binary_erosion(brain_outer, iterations=dura_thickness)
        
        # 4. Субарахноидальное пространство
        csf_thickness = max(2, int(3 * (h/140.0)))
        csf_space = binary_erosion(brain_inner, iterations=csf_thickness) ^ brain_inner
        brain_surface = binary_erosion(brain_inner, iterations=csf_thickness)
        
        # 5. Создаём полушария мозга с долями
        cortex_mask, sulci_mask, noise_field, white_matter = self.create_brain_hemispheres(brain_surface, shape)        
        
        # 6. Желудочковая система
        ventricles = self.create_ventricular_system(shape)
        ventricles &= white_matter
        
        # 7. Мозжечок
        cerebellum_mask = self.create_cerebellum(shape, white_matter)
        
        # 8. Ствол мозга
        brainstem_mask = self.create_brainstem(shape, white_matter)
        
        # 9. Опухоль
        if tumor_pos is not None:
            tumor_z, tumor_y, tumor_x = tumor_pos
            if not (0 <= tumor_z < d and 0 <= tumor_y < h and 0 <= tumor_x < w):
                print(f"   ⚠️ Позиция опухоли вне сетки! Генерация случайной...")
                tumor_pos = None
            elif not (white_matter[tumor_z, tumor_y, tumor_x] or cortex_mask[tumor_z, tumor_y, tumor_x]):
                print(f"   ⚠️ Позиция вне мозговой ткани! Коррекция...")
                valid_z, valid_y, valid_x = np.where(white_matter | cortex_mask)
                if len(valid_z) > 0:
                    dists = np.sqrt((valid_z - tumor_z)**2 + (valid_y - tumor_y)**2 + (valid_x - tumor_x)**2)
                    nearest_idx = np.argmin(dists)
                    tumor_z, tumor_y, tumor_x = valid_z[nearest_idx], valid_y[nearest_idx], valid_x[nearest_idx]
                    print(f"   ✅ Скорректированная позиция: Z={tumor_z}, Y={tumor_y}, X={tumor_x}")
                else:
                    tumor_pos = None
        
        if tumor_pos is None:
            # Ищем в белом веществе
            valid_z, valid_y, valid_x = np.where(white_matter)
            if len(valid_z) == 0:
                # Fallback: ищем в любой мозговой ткани (кора + белое вещество)
                brain_tissue = cortex_mask | white_matter
                valid_z, valid_y, valid_x = np.where(brain_tissue)
            
            if len(valid_z) > 0:
                idx = np.random.randint(0, len(valid_z))
                tumor_z, tumor_y, tumor_x = valid_z[idx], valid_y[idx], valid_x[idx]
                print(f"   🎯 Опухоль создана в позиции: Z={tumor_z}, Y={tumor_y}, X={tumor_x}")
            else:
                print("   ❌ Не удалось найти позицию для опухоли в мозге!")
                tumor_z, tumor_y, tumor_x = int(d*0.5), int(h*0.5), int(w*0.5)
        else:
            print(f"   🎯 Опухоль создана в заданной позиции: Z={tumor_z}, Y={tumor_y}, X={tumor_x}")
        
        # Маска опухоли
        dist_to_tumor = np.sqrt((z - tumor_z)**2 + (y - tumor_y)**2 + (x - tumor_x)**2)
        tumor_mask = (dist_to_tumor <= tumor_radius) & (white_matter | cortex_mask)
        
        # 10. Заполнение свойств
        eps_map = np.ones(shape, dtype=np.float32)
        cond_map = np.zeros(shape, dtype=np.float32)
        temp_map = np.ones(shape, dtype=np.float32) * 20.0
        
        def fill_tissue(mask, tissue_key):
            if np.any(mask):
                props = self.tissue_props[tissue_key]
                eps_map[mask] = np.clip(np.random.normal(props['mean_eps'], props['std_eps'], np.sum(mask)), 1.0, None)
                cond_map[mask] = np.clip(np.random.normal(props['mean_cond'], props['std_cond'], np.sum(mask)), 0.0, None)
                temp_map[mask] = props['temp_base']
        
        # Заполняем ткани в порядке от внешних к внутренним
        fill_tissue(head_mask & ~skull_compact_outer, 'scalp')  # Кожа
        fill_tissue(skull_compact_outer, 'skull_compact')  # Компактная кость (наружная)
        fill_tissue(skull_spongy, 'skull_spongy')  # Губчатая кость
        fill_tissue(skull_compact_inner, 'skull_compact')  # Компактная кость (внутренняя)
        fill_tissue(dura_mask, 'dura_mater')  # Твёрдая оболочка
        fill_tissue(csf_space, 'csf')  # Субарахноидальное пространство
        fill_tissue(cortex_mask & ~tumor_mask, 'gray_matter')  # Серое вещество
        fill_tissue(white_matter & ~ventricles & ~cerebellum_mask & ~brainstem_mask & ~tumor_mask, 'white_matter')  # Белое вещество
        fill_tissue(ventricles, 'csf')  # Желудочки
        fill_tissue(cerebellum_mask, 'cerebellum')  # Мозжечок
        fill_tissue(brainstem_mask, 'brainstem')  # Ствол мозга
        fill_tissue(tumor_mask, 'tumor')  # Опухоль
        
        # 11. Температурный градиент
        # Расстояние от поверхности черепа
        dist_from_surface = distance_transform_edt(~head_mask).astype(np.float32)
        dist_from_surface[~head_mask] = 0
        max_dist = dist_from_surface[head_mask].max()
        
        if max_dist > 0:
            # Температура повышается от поверхности (33°C) к центру (37°C)
            normalized_depth = dist_from_surface / max_dist
            temp_map += 4.0 * (normalized_depth ** 0.5) * head_mask
        
        # Вклад опухоли
        if np.any(tumor_mask):
            tumor_sigma = tumor_radius * 1.5
            temp_map += 2.0 * np.exp(-dist_to_tumor**2 / (2 * tumor_sigma**2)) * head_mask
            inflammation_radius = tumor_radius * 2.5
            temp_map += 0.8 * np.exp(-dist_to_tumor**2 / (2 * inflammation_radius**2)) * head_mask
        
        # Гладкость
        sigma = max(1.0, 1.0 * (h/140.0))
        temp_map = gaussian_filter(temp_map, sigma=sigma)
        eps_map = gaussian_filter(eps_map, sigma=sigma)
        cond_map = gaussian_filter(cond_map, sigma=sigma)
        
        # Обнуляем воздух
        temp_map[~head_mask] = 20.0
        eps_map[~head_mask] = 1.0
        cond_map[~head_mask] = 0.0
        temp_map = np.clip(temp_map, 33.0, 40.0)
        
        self.tumor_center = (tumor_z, tumor_y, tumor_x)
        
        print(f"   ✅ Время создания фантома: {time.time() - start_time:.2f} сек")
        print(f"   Вокселей в голове: {np.sum(head_mask):,}")
        print(f"   Вокселей в опухоли: {np.sum(tumor_mask):,}")
        
        # Сохраняем структуры для визуализации
        structures = {
            'scalp': head_mask & ~skull_compact_outer,
            'skull': skull_compact_outer | skull_spongy | skull_compact_inner,
            'dura': dura_mask,
            'csf_space': csf_space,
            'cortex': cortex_mask,
            'white_matter': white_matter,
            'ventricles': ventricles,
            'cerebellum': cerebellum_mask,
            'brainstem': brainstem_mask,
            'tumor': tumor_mask
        }
        
        return eps_map, cond_map, temp_map, head_mask, tumor_mask, structures
    
    def compute_emissivity_3d(self, eps_map, mask=None):
        """
        Расчет коэффициента излучения для 3D объема.
        """
        sqrt_eps = np.sqrt(np.maximum(eps_map, 1.0))
        gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
        emissivity_fresnel = 1.0 - gamma**2
        
        emissivity = 0.88 + 0.11 * (emissivity_fresnel - 0.5) / 0.5
        emissivity = np.clip(emissivity, 0.98, 0.99)
        
        np.random.seed(42)
        noise = np.random.normal(0, 0.015, emissivity.shape)
        if mask is not None:
            emissivity = np.clip(emissivity + noise * mask, 0.90, 0.99)
        else:
            emissivity = np.clip(emissivity + noise, 0.90, 0.99)
        
        return emissivity
    
    def compute_sensitivity_kernel_3d(self, mask, ant_pos_3d, skull_mask=None):
        """
        Вычисляет 3D ядро чувствительности для одной антенны.
        """
        d, h, w = mask.shape
        z, y, x = np.ogrid[:d, :h, :w]
        
        # Расстояния в вокселях
        dist_xy = np.sqrt((x - ant_pos_3d[2])**2 + (y - ant_pos_3d[1])**2)
        dist_z = np.abs(z - ant_pos_3d[0])
        
        # Параметры гауссианы
        sigma_xy = 12.0  # Латеральное разрешение
        sigma_z = 20.0   # Глубина проникновения
        
        # 3D Гауссиана
        kernel = np.exp(-(dist_xy**2) / (2 * sigma_xy**2) - (dist_z**2) / (2 * sigma_z**2))
        
        # Учет черепа как барьера
        if skull_mask is not None:
            # Коэффициент затухания в кости
            skull_attenuation = 0.7
            kernel *= np.where(skull_mask, skull_attenuation, 1.0)
        
        # Учитываем только ткань
        kernel *= mask
        
        # Нормировка ядра
        sum_k = np.sum(kernel)
        if sum_k > 0:
            kernel /= sum_k
        
        return kernel
    
    def forward_scan_3d(self, temp_map, eps_map, mask, scan_positions_3d, skull_mask=None):
        """
        Прямое сканирование: расчет яркостной температуры (Tb) для каждой антенны.
        """
        temp_kelvin = temp_map + 273.15
        emissivity = self.compute_emissivity_3d(eps_map, mask)
        
        measurements = []
        emissivity_avg = []
        
        print(f"   📡 Сканирование {len(scan_positions_3d)} антеннами...")
        for i, pos in enumerate(scan_positions_3d):
            kernel = self.compute_sensitivity_kernel_3d(mask, pos, skull_mask)
            
            # Формула: Tb = sum(kernel * emissivity * T_kelvin)
            emissivity_avg.append(np.sum(kernel * emissivity))
            measurements.append(np.sum(kernel * emissivity * temp_kelvin))
        
        return np.array(measurements), np.array(emissivity_avg)
    
    def reconstruct_3d(self, Tb_data, emissivity_avg, scan_positions_3d, shape, mask, skull_mask=None):
        """
        3D реконструкция методом обратного проецирования.
        """
        recon_kelvin = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        
        print(f"   🔄 Реконструкция 3D объема...")
        for i, pos in enumerate(scan_positions_3d):
            kernel = self.compute_sensitivity_kernel_3d(mask, pos, skull_mask)
            
            emissivity_corr = emissivity_avg[i] if emissivity_avg[i] > 0.5 else 0.95
            Tb_corrected = Tb_data[i] / emissivity_corr
            
            recon_kelvin += kernel * Tb_corrected
            weight_sum += kernel
        
        weight_sum[weight_sum == 0] = 1.0
        recon_celsius = (recon_kelvin / weight_sum) - 273.15
        
        # Clip от выбросов
        valid_data = recon_celsius[mask]
        if len(valid_data) > 0:
            min_t, max_t = np.percentile(valid_data, [2, 98])
            if max_t > min_t:
                recon_celsius = np.clip(recon_celsius, min_t - 1, max_t + 1)
        
        recon_celsius = np.clip(recon_celsius, 32.0, 42.0)
        
        # Сглаживание
        recon_celsius = gaussian_filter(recon_celsius, sigma=1.0)
        
        # Обнуляем воздух
        recon_celsius[~mask] = np.nan
        
        return recon_celsius