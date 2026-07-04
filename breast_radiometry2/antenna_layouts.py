import numpy as np

def planar_grid(shape, n_y=8, n_x=8, z_height=-15, coverage=0.6):
    """
    Плоская квадратная решетка (как сейчас).
    coverage: какую часть груди покрываем (0.6 = 60% от центра)
    """
    d, h, w = shape
    y_range = np.linspace(int(h*(0.5 - coverage/2)), int(h*(0.5 + coverage/2)), n_y, dtype=int)
    x_range = np.linspace(int(w*(0.5 - coverage/2)), int(w*(0.5 + coverage/2)), n_x, dtype=int)
    return [(z_height, int(y), int(x)) for y in y_range for x in x_range]


def circular_ring(shape, n_ant=24, radius=50, z_height=-5, center=None):
    """
    Антенны по кругу (кольцу) над грудью.
    """
    d, h, w = shape
    if center is None:
        center = (h // 2, w // 2)
    angles = np.linspace(0, 2*np.pi, n_ant, endpoint=False)
    positions = []
    for a in angles:
        y = int(center[0] + radius * np.cos(a))
        x = int(center[1] + radius * np.sin(a))
        positions.append((z_height, y, x))
    return positions


def hemispherical_array(shape, n_theta=6, n_phi=12, radius=70, center=None, air_buffer_z=25):
    """
    Полусферическая решетка — антенны окружают грудь сверху.
    Теперь антенны располагаются в диапазоне Z ∈ [0, air_buffer_z-1]
    """
    d, h, w = shape
    if center is None:
        center = (h // 2, w // 2)
    
    positions = []
    thetas = np.linspace(0.1, np.pi/2 - 0.1, n_theta)
    phis = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    for theta in thetas:
        for phi in phis:
            # Сферические координаты -> декартовы
            # Z теперь положительный и в диапазоне [0, air_buffer_z-1]
            z = int((air_buffer_z - 1) * (1 - np.cos(theta)))  # от 0 до air_buffer_z-1
            y = int(center[0] + radius * np.sin(theta) * np.cos(phi))
            x = int(center[1] + radius * np.sin(theta) * np.sin(phi))
            
            # Проверка границ
            if 0 <= z < d and 0 <= y < h and 0 <= x < w:
                positions.append((z, y, x))
    
    return positions


def dual_plane(shape, n_y=8, n_z=6, coverage=0.7, air_buffer_z=25, side='both'):
    """
    Две боковые плоскости антенн: слева и справа от груди.
    Антенны расположены на уровне центра груди по Z, 
    распределены по осям Y (высота) и Z (глубина).
    
    Параметры:
    - n_y: количество антенн по вертикали (ось Y)
    - n_z: количество антенн по глубине (ось Z)
    - coverage: какую часть груди покрываем по Y и Z (0.7 = 70%)
    - side: 'left' | 'right' | 'both' — какая сторона
    """
    d, h, w = shape
    
    # Центр груди
    center_z = air_buffer_z + (d - air_buffer_z) * 0.3
    center_y = h // 2
    
    # Границы груди по Y и Z (примерно)
    # Грудь занимает примерно 30-85% по Z и 15-85% по Y
    z_min = air_buffer_z
    z_max = int(d * 0.85)
    y_min = int(h * 0.15)
    y_max = int(h * 0.85)
    
    # Диапазоны для антенн (с учётом coverage)
    z_range = np.linspace(
        int(z_min + (z_max - z_min) * (0.5 - coverage/2)),
        int(z_min + (z_max - z_min) * (0.5 + coverage/2)),
        n_z, dtype=int
    )
    y_range = np.linspace(
        int(y_min + (y_max - y_min) * (0.5 - coverage/2)),
        int(y_min + (y_max - y_min) * (0.5 + coverage/2)),
        n_y, dtype=int
    )
    
    positions = []
    
    # Расстояние от центра груди до боковой плоскости антенн
    # Антенны должны быть чуть дальше границы груди
    side_offset = int(w * 0.02)  # 5% от ширины сетки
    
    # ========== ЛЕВАЯ ПЛОСКОСТЬ (малый X) ==========
    if side in ('left', 'both'):
        x_left = int(w * 0.5 - w * 0.45 - side_offset)  # Слева от груди
        x_left = max(0, x_left)
        
        for z in z_range:
            for y in y_range:
                if 0 <= z < d and 0 <= y < h and 0 <= x_left < w:
                    positions.append((int(z), int(y), x_left))
    
    # ========== ПРАВАЯ ПЛОСКОСТЬ (большой X) ==========
    if side in ('right', 'both'):
        x_right = int(w * 0.5 + w * 0.45 + side_offset)  # Справа от груди
        x_right = min(w - 1, x_right)
        
        for z in z_range:
            for y in y_range:
                if 0 <= z < d and 0 <= y < h and 0 <= x_right < w:
                    positions.append((int(z), int(y), x_right))
    
    return positions


def manual_positions(positions_list):
    """
    Ручное задание координат. Формат: список кортежей (z, y, x)
    """
    return [(int(z), int(y), int(x)) for z, y, x in positions_list]