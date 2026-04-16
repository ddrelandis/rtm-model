import numpy as np
from scipy.ndimage import gaussian_filter, binary_erosion, binary_dilation
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
import time

class BreastRadiometryModelReal:
    def __init__(self, freq_ghz=3.0, resolution_mm=2, birads_category='B', temp_vmin=None, temp_vmax=None):
        self.freq = freq_ghz * 1e9
        self.omega = 2 * np.pi * self.freq
        self.c = 3e8
        self.res = resolution_mm / 1000.0
        self.eps0 = 8.854e-12
        self.tumor_center = None
        self.birads_category = birads_category
        
        self.birads_density = {'A': (0.10, 0.25), 'B': (0.26, 0.50), 'C': (0.51, 0.75), 'D': (0.76, 0.90)}
        self.temp_vmin = temp_vmin if temp_vmin is not None else 33.0
        self.temp_vmax = temp_vmax if temp_vmax is not None else 40.0

        # Свойства тканей: {ID: {'eps': ..., 'cond': ..., 'k': ..., 'perf': ..., 'Qm': ...}}
        self.tissue_props = {
            1: {'eps': 5.0, 'cond': 0.08, 'k': 0.21, 'perf': 0.0005, 'Qm': 400, 'name': 'Жир подк.'},
            2: {'eps': 45.0, 'cond': 2.4, 'k': 0.45, 'perf': 0.0020, 'Qm': 1200, 'name': 'Железистая'},
            3: {'eps': 5.0, 'cond': 0.10, 'k': 0.21, 'perf': 0.0005, 'Qm': 400, 'name': 'Жир внутр.'},
            4: {'eps': 5.0, 'cond': 0.10, 'k': 0.21, 'perf': 0.0005, 'Qm': 400, 'name': 'Жир ретро'},
            5: {'eps': 30.0, 'cond': 1.3, 'k': 0.35, 'perf': 0.0010, 'Qm': 800, 'name': 'Соединительная'},
            6: {'eps': 48.0, 'cond': 2.8, 'k': 0.45, 'perf': 0.0020, 'Qm': 1200, 'name': 'Протоки'},
            7: {'eps': 45.0, 'cond': 2.4, 'k': 0.45, 'perf': 0.0020, 'Qm': 1200, 'name': 'Дольки'},
            8: {'eps': 45.0, 'cond': 2.6, 'k': 0.35, 'perf': 0.0015, 'Qm': 1000, 'name': 'Сосок'},
            9: {'eps': 45.0, 'cond': 2.6, 'k': 0.35, 'perf': 0.0015, 'Qm': 1000, 'name': 'Ареола'},
            10: {'eps': 35.0, 'cond': 1.0, 'k': 0.25, 'perf': 0.0010, 'Qm': 800, 'name': 'Кожа'},
            11: {'eps': 50.0, 'cond': 2.0, 'k': 0.45, 'perf': 0.0025, 'Qm': 1500, 'name': 'Грудная стенка'},
            12: {'eps': 60.0, 'cond': 4.5, 'k': 0.50, 'perf': 0.0050, 'Qm': 6000, 'name': 'Опухоль'}
        }

    def solve_pennes_bioheat(self, breast_mask, tissue_type_map, T_blood=37.0, T_skin=34.0):
        print("🔥 Расчет температурного поля (Ур. Пеннеса)...")
        h, w = breast_mask.shape
        N = h * w
        dx = self.res
        dx2 = dx**2
        
        # Векторизуем свойства
        props_k = np.zeros(N)
        props_perf = np.zeros(N)
        props_qm = np.zeros(N)
        
        t_flat = tissue_type_map.ravel()
        mask_flat = breast_mask.ravel()
        
        for t_id, props in self.tissue_props.items():
            idx = t_flat == t_id
            props_k[idx] = props['k']
            props_perf[idx] = props['perf']
            props_qm[idx] = props['Qm']

        # Формируем диагонали для разреженной матрицы
        # Уравнение: k * Laplacian(T) + perf*(Ta - T) + Qm = 0
        # -> T * (4k + perf*dx2) - sum(T_neighbors*k) = perf*Ta*dx2 + Qm*dx2
        
        main_diag = 4.0 * props_k + props_perf * dx2
        off_diag = -props_k[:-1] # Соседи по X и Y будут усреднены или использованы отдельно
        # Для простоты используем изотропный шаблон
        diag_x = -0.5 * (props_k[:-1] + props_k[1:])
        diag_y = -0.5 * (props_k[:-w] + props_k[w:])
        
        main_diag[~mask_flat] = 1.0
        off_diag = np.zeros(N-1)
        
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(N, N), format='csc')
        # Заполняем Y-связи и X-связи вручную для точности, или используем упрощённый вариант
        # Для надёжности вернёмся к циклу, но с правильными знаками (он вызывается 1 раз)
        
        A = np.zeros((N, N))
        b = np.zeros(N)
        
        for y in range(h):
            for x in range(w):
                i = y * w + x
                if not breast_mask[y, x]:
                    A[i, i] = 1.0; b[i] = 20.0; continue
                    
                k = props_k[i]; perf = props_perf[i]; qm = props_qm[i]
                A[i, i] = 4*k + perf*dx2
                b[i] = perf*T_blood*dx2 + qm*dx2
                
                if y > 0: A[i, i-w] = -k
                if y < h-1: A[i, i+w] = -k
                if x > 0: A[i, i-1] = -k
                if x < w-1: A[i, i+1] = -k
                
                is_skin = (y==0 or x==0 or x==w-1) or (not breast_mask[max(0,y-1),x] or not breast_mask[min(h-1,y+1),x] or not breast_mask[y,max(0,x-1)] or not breast_mask[y,min(w-1,x+1)])
                if is_skin:
                    A[i, i] = 1.0; b[i] = T_skin
                elif y >= h-3: # Грудная стенка
                    A[i, i] = 1.0; b[i] = T_blood
                    
        return np.linalg.solve(A, b).reshape((h, w))

    def compute_wave_kernel(self, eps_map, cond_map, mask, ant_pos):
        h, w = mask.shape
        N = h * w
        k0 = self.omega / self.c
        dx = self.res; dx2 = dx**2
        
        eps_c = (eps_map - 1j * cond_map / (self.omega * self.eps0)).ravel()
        mask_flat = mask.ravel()
        
        # Быстрая сборка через diags
        main = -4.0/dx2 + k0**2 * eps_c
        off_x = np.ones(N-1)/dx2
        off_y = np.ones(N-w)/dx2
        
        A = diags([off_x, main, off_x, off_y, off_y], [-1, 0, 1, -w, w], shape=(N, N), format='csc')
        
        bg = np.where(~mask_flat)[0]
        if len(bg) > 0:
            A[bg, :] = 0; A[bg, bg] = 1.0
            
        src = np.zeros(N, dtype=complex)
        src_idx = ant_pos[0]*w + ant_pos[1]
        if mask_flat[src_idx]: src[src_idx] = 1.0+0j
        else:
            tissue = np.where(mask_flat)[0]
            if len(tissue) > 0: src[np.argmin(np.abs(tissue - src_idx))] = 1.0+0j
            
        try:
            E = spsolve(A, src).reshape((h, w))
            K = cond_map * (np.abs(E)**2) * mask
            K_sum = np.sum(K)
            return K / K_sum if K_sum > 1e-9 else self._quasi_static_kernel(mask, ant_pos)
        except Exception:
            return self._quasi_static_kernel(mask, ant_pos)

    def _quasi_static_kernel(self, mask, ant_pos):
        h, w = mask.shape
        y, x = np.ogrid[:h, :w]
        dx_m = (x - ant_pos[1]) * self.res
        dy_m = (y - ant_pos[0]) * self.res
        r_m = np.sqrt(dx_m**2 + dy_m**2)
        r_safe = np.maximum(r_m, self.res)
        # Физически корректное затухание на 3 ГГц
        alpha = self.omega / self.c * np.sqrt(25/2 * (np.sqrt(1 + (2.0/(self.omega*self.eps0*25))**2)-1))
        K = np.exp(-2*alpha*r_safe) / (r_safe**2 + 1e-6) * mask
        return K / np.sum(K)

    def compute_sensitivity_kernel(self, mask, ant_pos):
        if self.eps_map is None: raise ValueError("Сначала создайте фантом")
        return self.compute_wave_kernel(self.eps_map, self.cond_map, mask, ant_pos)

    def create_anatomical_phantom(self, shape=(160, 200), tumor_radius=12, tumor_pos=None):
        start_time = time.time()
        h, w = shape
        y, x = np.ogrid[:h, :w]
        center_x = w / 2.0; scale_factor = h / 80.0
        top_y = int(h * 0.12)
        breast_mask = np.zeros(shape, dtype=bool)
        for yi in range(top_y, h):
            norm_y = (yi - top_y) / (h - top_y)
            wf = w*0.06 if norm_y<0.25 else (w*0.35 if norm_y<0.6 else w*0.55)
            if 0.25 <= norm_y < 0.6: wf = w*0.35 + (w*0.55-w*0.35)*((norm_y-0.25)/0.35)
            elif norm_y < 0.25: wf = w*0.06 + (w*0.35-w*0.06)*(norm_y/0.25)**0.5
            xl, xr = max(0, int(center_x - wf)), min(w, int(center_x + wf))
            breast_mask[yi, xl:xr] = True
        breast_mask = binary_dilation(breast_mask, iterations=max(1, int(2*scale_factor)))
        breast_mask = gaussian_filter(breast_mask.astype(float), sigma=0.8*scale_factor) > 0.5
        
        # Структуры
        nc_y, nc_x = int(h*0.15), int(w/2)
        areola_m = ((x-nc_x)**2 + (y-nc_y)**2 <= int(w*0.10)**2) & breast_mask
        nipple_m = ((x-nc_x)**2 + (y-nc_y)**2 <= int(w*0.04)**2) & areola_m
        skin_m = (binary_erosion(breast_mask, iterations=max(2, int(2*scale_factor))) ^ breast_mask) & breast_mask
        subcut_m = (binary_erosion(binary_erosion(breast_mask, iterations=max(2, int(2*scale_factor))), iterations=max(4, int(h*0.06))) ^ binary_erosion(breast_mask, iterations=max(2, int(2*scale_factor)))) & breast_mask & ~skin_m
        retro_m = (y >= int(h*0.65)) & breast_mask & ~skin_m & ~subcut_m
        gland_m = breast_mask & ~skin_m & ~subcut_m & ~retro_m & ~areola_m
        body_m = (y >= int(h*0.75)) & breast_mask
        
        # Железистая ткань
        density_range = self.birads_density[self.birads_category]
        target_gland = np.random.uniform(density_range[0], density_range[1])
        lobe_m, duct_m, conn_m = np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
        gc_y, gc_x = int(h*0.45), int(w/2)
        for i in range(np.random.randint(15, 21)):
            ang = (2*np.pi*i)/21
            lw = np.random.uniform(0.15, 0.25)*np.pi/21
            dy, dx = y-gc_y, x-gc_x
            am = np.arctan2(dy, dx)
            ad = np.minimum(np.abs(am-ang), 2*np.pi-np.abs(am-ang))
            lobe_m |= (ad < lw) & gland_m & (np.sqrt(dx**2+dy**2)<w*0.4)
            
        li = np.where(lobe_m)
        for _ in range(int(np.sum(lobe_m)*0.003)):
            if len(li[0])==0: break
            idx = np.random.randint(0, len(li[0]))
            cy, cx = li[0][idx], li[1][idx]
            ls = max(4, int(np.random.randint(4,10)*scale_factor))
            yy, xx = np.ogrid[:h, :w]
            lobe_m |= ((xx-cx)**2 + (yy-cy)**2 <= ls**2) & lobe_m
            
        for i in range(min(12, 18)):
            ang = (2*np.pi*i)/18 + np.random.uniform(-0.2, 0.2)
            t = np.linspace(0,1,100)
            for ti in t:
                px, py = int(nc_x + ti*(w*0.35)*np.cos(ang)), int(nc_y + ti*(h*0.4)*np.sin(ang))
                if 0<=px<w and 0<=py<h: duct_m |= ((x-px)**2+(y-py)**2 <= max(3,int(3*scale_factor))**2) & gland_m
                
        for _ in range(max(60, int(60*scale_factor))):
            fy, fx = np.random.randint(top_y, h), np.random.randint(int(center_x-w*0.55), int(center_x+w*0.55))
            fl = max(10, int(np.random.randint(10,25)*scale_factor))
            fa = np.random.uniform(0, 2*np.pi)
            for l in range(fl):
                px, py = int(fx+l*np.cos(fa)), int(fy+l*np.sin(fa))
                if 0<=px<w and 0<=py<h: conn_m |= ((x-px)**2+(y-py)**2 <= max(2,int(2*scale_factor))**2) & gland_m

        ag = np.sum(gland_m)
        tg = int(ag * target_gland)
        gp = np.zeros(shape); gp[lobe_m]=3; gp[duct_m]=2; gp[gland_m]=1
        gi = np.where(gland_m & (gp>0))
        fg_m = np.zeros(shape, dtype=bool)
        if len(gi[0])>0:
            pri = gp[gi]; si = np.argsort(-pri)
            for idx in si[:tg]: fg_m[gi[0][idx], gi[1][idx]] = True
        ig_m = gland_m & ~fg_m

        self.eps_map, self.cond_map = np.zeros(shape), np.zeros(shape)
        self.tissue_type_map = np.zeros(shape, dtype=int)
        
        def fill(mask, tid):
            if np.any(mask):
                p = self.tissue_props[tid]
                self.eps_map[mask] = np.clip(np.random.normal(p['eps'], p['eps']*0.1), 1.0, None)
                self.cond_map[mask] = np.clip(np.random.normal(p['cond'], p['cond']*0.1), 0.01, None)
                self.tissue_type_map[mask] = tid
            return mask

        fill(subcut_m, 1); fill(fg_m, 2); fill(ig_m, 3); fill(retro_m, 4)
        fill(conn_m, 5); fill(duct_m, 6); fill(lobe_m & fg_m, 7)
        if np.any(areola_m): fill(areola_m, 9)
        if np.any(nipple_m): fill(nipple_m, 8)
        if np.any(skin_m): fill(skin_m, 10)
        if np.any(body_m): fill(body_m, 11)
        self.breast_mask = breast_mask
        self.eps_map[~breast_mask] = 1.0; self.cond_map[~breast_mask] = 0.0

        # Опухоль
        self.tumor_center = None
        ty, tx = None, None
        if tumor_pos is not None and fg_m[tumor_pos[0], tumor_pos[1]]:
            ty, tx = tumor_pos
        else:
            vy, vx = np.where(fg_m & (y>h*0.3) & (y<h*0.65))
            if len(vy)>0: idx=np.random.randint(0,len(vy)); ty,tx=vy[idx],vx[idx]
            
        if ty is not None:
            self.tumor_center = (ty, tx)
            y_t, x_t = np.ogrid[:h, :w]
            tm = np.sqrt((x_t-tx)**2 + (y_t-ty)**2) <= tumor_radius
            self.tissue_type_map[tm] = 12
            p = self.tissue_props[12]
            self.eps_map[tm] = np.random.normal(p['eps'], p['eps']*0.1)
            self.cond_map[tm] = np.random.normal(p['cond'], p['cond']*0.1)
            print(f"✅ Опухоль: Y={ty}, X={tx}")

        temp_map = self.solve_pennes_bioheat(breast_mask, self.tissue_type_map)
        temp_map[~breast_mask] = 20.0
        print(f"⏱️ Фантом готов за {time.time()-start_time:.2f} сек")
        return self.eps_map, self.cond_map, temp_map, breast_mask, areola_m, nipple_m, body_m, self.tissue_type_map

    def compute_emissivity(self, eps_map, mask=None):
        sqrt_eps = np.sqrt(np.maximum(eps_map, 1.0))
        gamma = (sqrt_eps - 1.0) / (sqrt_eps + 1.0)
        emissivity = 0.85 + 0.13 * (1.0 - gamma**2)
        return np.clip(emissivity, 0.90, 0.99)

    def forward_scan(self, eps_map, cond_map, temp_map, mask, scan_positions):
        measurements, emissivity_avg = [], []
        temp_kelvin = temp_map + 273.15
        emissivity_map = self.compute_emissivity(eps_map, mask)
        
        for pos in scan_positions:
            K = self.compute_sensitivity_kernel(mask, pos)
            Tb = np.sum(K * emissivity_map * temp_kelvin)
            measurements.append(Tb)
            emissivity_avg.append(np.sum(K * emissivity_map))
        return np.array(measurements), np.array(emissivity_avg)

    def reconstruct_simple(self, measurements, emissivity_avg, scan_positions, shape, mask):
        recon_kelvin = np.zeros(shape); weight_sum = np.zeros(shape)
        for i, pos in enumerate(scan_positions):
            kernel = self.compute_sensitivity_kernel(mask, pos)
            e_corr = emissivity_avg[i] if emissivity_avg[i] > 0.5 else 0.95
            Tb_corr = measurements[i] / e_corr
            recon_kelvin += kernel * Tb_corr
            weight_sum += kernel
        with np.errstate(divide='ignore', invalid='ignore'):
            recon_kelvin = np.where(weight_sum > 1e-12, recon_kelvin / weight_sum, np.nan)
        recon_celsius = recon_kelvin - 273.15
        recon_celsius = gaussian_filter(np.where(mask, recon_celsius, 0), sigma=1.0)
        return np.where(mask, recon_celsius, np.nan)