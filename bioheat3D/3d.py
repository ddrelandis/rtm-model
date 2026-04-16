import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from time import time

# ========================
# 1. ПАРАМЕТРЫ СЕТКИ И ВРЕМЕНИ
# ========================
Nx, Ny, Nz = 40, 40, 30          # число узлов по осям
Lx, Ly, Lz = 0.12, 0.12, 0.09    # размеры модели [м]
dx, dy, dz = Lx/Nx, Ly/Ny, Lz/Nz # шаги сетки
dt = 5.0                         # шаг по времени [с]
t_final = 600.0                  # время моделирования [с]
Nt = int(t_final / dt)

# ========================
# 2. ГЕОМЕТРИЯ И ТИПЫ ТКАНЕЙ
# ========================
x = np.linspace(-Lx/2, Lx/2, Nx)
y = np.linspace(-Ly/2, Ly/2, Ny)
z = np.linspace(0, Lz, Nz)       # z=0 - грудная стенка, z=Lz - кожа
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Простая эллипсоидная модель молочной железы
a, b, c_geo = 0.06, 0.06, 0.04
mask_breast = (X/a)**2 + (Y/b)**2 + ((Z-0.05)/c_geo)**2 <= 1.0

# Типы тканей: 0=фон/воздух, 1=жировая, 2=железистая, 3=кожа
tissue = np.zeros((Nx, Ny, Nz), dtype=int)
tissue[mask_breast] = 1

# Внутренний железистый "диск"
mask_gland = (X/0.03)**2 + (Y/0.03)**2 + ((Z-0.06)/0.025)**2 <= 1.0
tissue[mask_breast & mask_gland] = 2

# Поверхностный слой кожи (z ≈ Lz)
skin_thickness = 0.002
mask_skin = (Z >= Lz - skin_thickness) & mask_breast
tissue[mask_skin] = 3
k_cond = np.zeros_like(tissue, dtype=float) 
# ========================
# 3. ТЕПЛОФИЗИЧЕСКИЕ ПАРАМЕТРЫ
# ========================
rho   = np.zeros_like(tissue, dtype=float)
c     = np.zeros_like(tissue, dtype=float)
k     = np.zeros_like(tissue, dtype=float)
omega = np.zeros_like(tissue, dtype=float)
Q_met = np.zeros_like(tissue, dtype=float)

# Жировая (1)
rho[tissue==1], c[tissue==1], k[tissue==1] = 930.0, 2400.0, 0.21
omega[tissue==1], Q_met[tissue==1] = 0.0015, 450.0

# Железистая (2)
rho[tissue==2], c[tissue==2], k[tissue==2] = 1050.0, 3500.0, 0.50
omega[tissue==2], Q_met[tissue==2] = 0.0050, 1100.0

# Кожа (3)
rho[tissue==3], c[tissue==3], k[tissue==3] = 1100.0, 3400.0, 0.39
omega[tissue==3], Q_met[tissue==3] = 0.0020, 600.0

# Фон/воздух (0) - задаём малые значения для численной устойчивости
rho[tissue==0], c[tissue==0], k[tissue==0] = 1.0, 1.0, 1e-6
omega[tissue==0], Q_met[tissue==0] = 0.0, 0.0

# Параметры крови
rho_b, c_b, T_a = 1050.0, 3617.0, 37.0

# ========================
# 4. ГРАНИЧНЫЕ УСЛОВИЯ (Robin на коже)
# ========================
h_conv = 10.0          # Вт/(м²·К)
T_env = 25.0           # °C
eps = 0.96
sigma = 5.67e-8
# Линеаризация излучения вокруг T_env (стандартный приём)
h_rad = 4 * eps * sigma * (T_env + 273.15)**3
h_total = h_conv + h_rad

# Маска кожи и направление нормали (для z=Lz нормаль = +k)
is_skin = (tissue == 3) & (Z >= Lz - skin_thickness/2)

# ========================
# 5. СБОРКА МАТРИЦЫ A и ВЕКТОРА b (неявная схема Эйлера)
# ========================
N = Nx * Ny * Nz
rows, cols, data = [], [], []

def idx(i, j, k): return i * Ny * Nz + j * Nz + k
def is_inside(i, j, k): return 0 <= i < Nx and 0 <= j < Ny and 0 <= k < Nz

# Направления соседей: (di, dj, dk, d_len)
neighbors = [(1,0,0,dx), (-1,0,0,dx), (0,1,0,dy), (0,-1,0,dy), (0,0,1,dz), (0,0,-1,dz)]

for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            n = idx(i, j, k)
            
            storage = rho[i,j,k] * c[i,j,k] / dt
            perf = omega[i,j,k] * rho_b * c_b
            A_diag = storage + perf
            b_val = perf * T_a + Q_met[i,j,k]

            k_center = k_cond[i,j,k]  # ← ИСПОЛЬЗУЕМ ПРАВИЛЬНОЕ ИМЯ
            
            for di, dj, dk, d_len in neighbors:
                ni, nj, nk = i+di, j+dj, k+dk
                if is_inside(ni, nj, nk):
                    k_half = (k_center + k_cond[ni,nj,nk]) / 2.0
                    coeff = k_half / (d_len**2)
                    A_diag += coeff
                    rows.append(n)
                    cols.append(idx(ni, nj, nk))
                    data.append(-coeff)  # Внедиагональный элемент отрицательный
                # else: граница домена (по умолчанию адиабата, обработка ниже)

            # Robin BC на коже (z ≈ Lz)
            if is_skin[i,j,k]:
                k_skin = k_cond[i,j,k]
                A_diag = storage + perf + h_total / dz
                b_val = perf * T_a + Q_met[i,j,k] + (h_total / dz) * T_env

            rows.append(n)
            cols.append(n)
            data.append(A_diag)  # Диагональный элемент

# Формируем вектор правой части
b = np.zeros(N)
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            n = idx(i,j,k)
            perf = omega[i,j,k] * rho_b * c_b
            val = perf * T_a + Q_met[i,j,k]
            if is_skin[i,j,k]:
                val += (h_total / dz) * T_env
            b[n] = val

A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsr()

# ========================
# 6. ВРЕМЕННОЙ ЦИКЛ
# ========================
T = np.full((Nx, Ny, Nz), T_a)  # начальное условие
T_flat = T.ravel()

print(f"Начало расчёта: {N} узлов, {Nt} шагов...")
t0 = time()
for step in range(Nt):
    T_flat = spla.spsolve(A, b + (rho*c/dt).ravel() * T_flat)
    if step % 10 == 0 or step == Nt-1:
        elapsed = time() - t0
        print(f"Шаг {step}/{Nt} | t={step*dt:.0f}s | Time={elapsed:.2f}s")

T = T_flat.reshape((Nx, Ny, Nz))
print(f"Расчёт завершён. Общее время: {time()-t0:.2f} с")

# ========================
# 7. ВИЗУАЛИЗАЦИЯ
# ========================
plt.figure(figsize=(10, 4))
# Срез по центру Y
plt.subplot(121)
slice_y = Ny//2
cmap = plt.get_cmap('turbo')
c = plt.imshow(T[:, slice_y, :].T, origin='lower', cmap=cmap, 
               extent=[x[0]*100, x[-1]*100, z[0]*100, z[-1]*100])
plt.colorbar(c, label='Temperature [°C]')
plt.xlabel('X [cm]')
plt.ylabel('Z [cm]')
plt.title('X-Z slice (Y=mid)')

# Распределение по глубине в центре
plt.subplot(122)
plt.plot(z*100, T[Nx//2, Ny//2, :], 'o-', markersize=4)
plt.axhline(T_env, color='k', ls='--', label='T_env')
plt.axhline(T_a, color='r', ls='--', label='T_art')
plt.xlabel('Depth Z [cm]')
plt.ylabel('Temperature [°C]')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()