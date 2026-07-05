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


def dual_plane(shape, n_y=6, n_x=6, z_top=-5, z_bottom=70, coverage=0.6):
    """
    Две параллельные плоскости антенн: сверху и снизу (имитация компрессии).
    """
    top = planar_grid(shape, n_y, n_x, z_top, coverage)
    d, h, w = shape
    # Для нижней плоскости инвертируем Z
    bottom = [(-z_bottom, pos[1], pos[2]) for pos in planar_grid(shape, n_y, n_x, -z_top, coverage)]
    return top + bottom


def manual_positions(positions_list):
    """
    Ручное задание координат. Формат: список кортежей (z, y, x)
    """
    return [(int(z), int(y), int(x)) for z, y, x in positions_list]