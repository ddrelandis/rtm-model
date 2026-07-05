"""
Интерактивный редактор позиций и направлений антенн на 3D-модели МРТ.
Позволяет кликать на поверхность груди и расставлять антенны мышкой.
"""
import numpy as np
import pyvista as pv
import os
import json
import subprocess

DATA_DIR = "data_real"
CONFIG_FILE = os.path.join(DATA_DIR, "antenna_config.npy")


class AntennaEditor:
    """Интерактивный редактор антенн на 3D-модели молочной железы."""
    
    # Предустановленные направления
    DIRECTIONS = {
        'down':    np.array([0, 0, 1], dtype=np.float32),    # Вглубь (по +Z PyVista)
        'up':      np.array([0, 0, -1], dtype=np.float32),   # Наружу
        'left':    np.array([-1, 0, 0], dtype=np.float32),   # Влево
        'right':   np.array([1, 0, 0], dtype=np.float32),    # Вправо
        'front':   np.array([0, -1, 0], dtype=np.float32),   # Вперёд
        'back':    np.array([0, 1, 0], dtype=np.float32),    # Назад
    }
    DIRECTION_NAMES = list(DIRECTIONS.keys())
    
    def __init__(self):
        # Загрузка данных
        print("📂 Загрузка данных...")
        self.temp_map = np.load(os.path.join(DATA_DIR, "real_temp_map.npy"))
        self.breast_mask = np.load(os.path.join(DATA_DIR, "real_breast_mask.npy"))
        self.eps_map = np.load(os.path.join(DATA_DIR, "real_eps_map.npy"))
        print(f"   ✅ Загружен объем {self.temp_map.shape}")
        
        # Список антенн: [(x, y, z, direction_name), ...]
        self.antennas = []
        self.current_direction_idx = 0  # индекс в DIRECTION_NAMES
        
        # Подготовка PyVista сетки
        self._prepare_grid()
        
    def _prepare_grid(self):
        """Подготавливает PyVista ImageData из массивов"""
        # Транспонируем (z,y,x) -> (x,y,z) для PyVista
        temp_grid = self.temp_map.transpose(2, 1, 0).astype(np.float32)
        breast_grid = self.breast_mask.transpose(2, 1, 0).astype(np.uint8)
        
        self.grid = pv.ImageData(dimensions=temp_grid.shape)
        self.grid.point_data["temp"] = temp_grid.flatten(order="F")
        self.grid.point_data["breast"] = breast_grid.flatten(order="F")
        
        # Создаём поверхность груди для point picking
        self.breast_surface = self.grid.contour(
            isosurfaces=[0.5], 
            scalars="breast"
        )
        print(f"   ✅ Поверхность груди: {self.breast_surface.n_points} точек")
            
    def run(self):
        """Запускает интерактивный редактор (упрощенная версия без emoji)"""
        self.plotter = pv.Plotter(window_size=[1600, 1000])
        self.plotter.set_background("black")
        
        # 1. Полупрозрачный контур груди (с явным pickable=True)
        self.plotter.add_mesh(
            self.breast_surface,
            color="peachpuff",
            opacity=0.4,  # Увеличиваем непрозрачность для лучшего picking
            name="breast_surface",
            pickable=True  # ← ВАЖНО: делаем поверхность "кликабельной"
        )
        
        # 2. Срез с температурой (фон)
        bounds = self.grid.bounds
        z_mid = (bounds[4] + bounds[5]) / 2
        slice_z = self.grid.slice(normal='z', origin=(0, 0, z_mid))
        self.plotter.add_mesh(
            slice_z, scalars="temp", cmap="jet",
            clim=[30.0, 42.0], opacity=0.6,  # ← Расширяем диапазон
            name="temp_slice",
            pickable=False  # Срез не должен перехватывать клики
        )
        
        # 3. Используем простой observer + pick_mouse_position
        self.plotter.iren.add_observer(
            "LeftButtonPressEvent", 
            self._mouse_click_handler_simple
        )
        
        # 4. Привязываем клавиши
        self.plotter.add_key_event('u', self._undo_last_antenna)
        self.plotter.add_key_event('d', self._cycle_direction)
        self.plotter.add_key_event('s', self._save_config)
        self.plotter.add_key_event('r', self._run_calculation)
        self.plotter.add_key_event('c', self._clear_all)
        
        # 5. Инструкции БЕЗ emoji (только ASCII)
        instructions_text = (
            "LMB on breast = add antenna\n"
            "d = change direction\n"
            "u = undo last\n"
            "c = clear all\n"
            "s = save config\n"
            "r = run calculation"
        )
        self.plotter.add_text(
            instructions_text,
            position="lower_left",
            font_size=10,
            color="white",
            name="instructions",
            font="courier"  # Моноширинный шрифт для выравнивания
        )
        
        # Заголовок БЕЗ emoji
        self._update_direction_indicator()
        
        self.plotter.add_axes()
        print("\n" + "=" * 60)
        print("INTERACTIVE ANTENNA EDITOR")
        print("=" * 60)
        print(f"   Current direction: {self.DIRECTION_NAMES[self.current_direction_idx].upper()}")
        print("   Left-click on breast surface to add antennas")
        print("=" * 60 + "\n")
        
        self.plotter.show()


    def _mouse_click_handler_simple(self, obj, event):
        """
        Упрощенный обработчик кликов мыши.
        Использует pick_mouse_position() — работает на 100%.
        """
        # Получаем 3D-точку под курсором
        picked_point = self.plotter.pick_mouse_position()
        
        if picked_point is None:
            return
        
        picked_point = np.array(picked_point)
        
        # Находим ближайшую точку на поверхности груди
        # Это "притягивает" клик к поверхности
        closest_idx = self.breast_surface.find_closest_point(picked_point)
        surface_point = self.breast_surface.points[closest_idx]
        
        # Проверяем расстояние (клик должен быть близко к поверхности)
        dist = np.linalg.norm(picked_point - surface_point)
        
        # Порог близости (в единицах координат PyVista)
        # Если кликнул далеко от поверхности — игнорируем
        if dist < 30.0:  # ← Подстрой под свой масштаб (10-50)
            self._on_point_picked(surface_point)
        else:
            print(f"   [info] Click too far from surface (dist={dist:.1f})")


    def _update_direction_indicator(self):
        """Обновляет индикатор направления БЕЗ emoji"""
        current_dir = self.DIRECTION_NAMES[self.current_direction_idx]
        
        # Удаляем старый заголовок
        try:
            self.plotter.remove_actor("title_actor")
        except:
            pass
        
        # Текст БЕЗ emoji
        title = (f"ANTENNA EDITOR | "
                f"Count: {len(self.antennas)} | "
                f"Direction: {current_dir.upper()}")
        
        self.plotter.add_text(
            title, 
            position="upper_edge",
            font_size=12, 
            color="white", 
            name="title_actor",
            font="arial"
        )
    
    def _on_point_picked(self, picked_point):
        """
        Callback при клике на surface.
        picked_point: координаты точки на поверхности (x, y, z) в PyVista-координатах
        """
        if picked_point is None:
            return
        
        x, y, z = picked_point
        direction_name = self.DIRECTION_NAMES[self.current_direction_idx]
        
        # Добавляем антенну в список
        self.antennas.append({
            'pos': (float(x), float(y), float(z)),
            'direction': direction_name,
        })
        
        print(f"   ✅ Добавлена антенна #{len(self.antennas)}: "
              f"pos=({x:.1f}, {y:.1f}, {z:.1f}), dir={direction_name.upper()}")
        
        # Перерисовываем антенны
        self._redraw_antennas()
    
    def _redraw_antennas(self):
        """Перерисовывает все антенны (сферы + стрелки)"""
        # Удаляем старые (если были)
        for name in list(self.plotter.renderer.actors.keys()):
            if name.startswith("antenna_") or name.startswith("arrow_"):
                self.plotter.remove_actor(name)
        
        if len(self.antennas) == 0:
            return
        
        # Собираем массивы точек и направлений
        points = []
        directions = []
        
        for i, ant in enumerate(self.antennas):
            points.append(ant['pos'])
            dir_name = ant['direction']
            directions.append(self.DIRECTIONS[dir_name])
        
        points = np.array(points, dtype=np.float32)
        directions = np.array(directions, dtype=np.float32)
        
        # 1. Сферы (позиции антенн)
        self.plotter.add_points(
            points,
            color="cyan",
            point_size=10,
            render_points_as_spheres=True,
            name="antenna_points"
        )
        
        # 2. Стрелки (направления)
        # mag = длина стрелки (подбери под свой масштаб!)
        self.plotter.add_arrows(
            points,
            directions,
            mag=8.0,
            color="lime",
            opacity=0.9,
            name="antenna_arrows"
        )
        
        # 3. Номера антенн (опционально)
        labels = [str(i + 1) for i in range(len(self.antennas))]
        label_points = pv.PolyData(points)
        label_points["labels"] = labels
        self.plotter.add_point_labels(
            label_points, "labels",
            font_size=12, text_color="yellow",
            shape=None,  # без фона
            show_points=False,
            name="antenna_labels"
        )
        
        self.plotter.render()
    
    def _undo_last_antenna(self):
        """Отменяет последнюю добавленную антенну"""
        if self.antennas:
            removed = self.antennas.pop()
            print(f"   ↩️  Удалена антенна: pos={removed['pos']}, "
                  f"dir={removed['direction'].upper()}")
            self._redraw_antennas()
        else:
            print("   ⚠️  Нечего отменять")
    
    def _cycle_direction(self):
        """Переключает текущее направление для новых антенн"""
        self.current_direction_idx = (self.current_direction_idx + 1) % len(self.DIRECTION_NAMES)
        new_dir = self.DIRECTION_NAMES[self.current_direction_idx]
        print(f"   🧭 Текущее направление: {new_dir.upper()}")
        self._update_direction_indicator()
    
    def _update_direction_indicator(self):
        """Обновляет индикатор текущего направления в заголовке"""
        current_dir = self.DIRECTION_NAMES[self.current_direction_idx]
        title = (f"📡 Редактор антенн | "
                f"Всего: {len(self.antennas)} | "
                f"Направление: {current_dir.upper()}")
        
        # Удаляем старый заголовок (если был)
        try:
            self.plotter.remove_actor("title_actor")
        except:
            pass
        
        # ✅ ИСПРАВЛЕНИЕ: используем add_text с name для возможности обновления
        self.plotter.add_text(
            title, 
            position="upper_edge",
            font_size=12, 
            color="white", 
            name="title_actor"  # ← теперь можем удалять и перерисовывать
        )
    
    def _clear_all(self):
        """Удаляет все антенны"""
        n = len(self.antennas)
        self.antennas.clear()
        print(f"   🗑  Очищено {n} антенн")
        self._redraw_antennas()
        self._update_direction_indicator()
    
    def _save_config(self):
        """Сохраняет конфигурацию антенн в файл"""
        if not self.antennas:
            print("   ⚠️  Нет антенн для сохранения")
            return
        
        # Конвертируем в формат для scan_real_3d.py
        # Формат: список dict с pos (в PyVista-координатах) и direction (вектор)
        config = {
            'antennas': [],
            'created': str(np.datetime64('now')),
            'count': len(self.antennas),
        }
        
        for ant in self.antennas:
            config['antennas'].append({
                'pos_pyvista': ant['pos'],  # (x, y, z) в координатах PyVista
                'direction_name': ant['direction'],
                'direction_vector': self.DIRECTIONS[ant['direction']].tolist(),
            })
        
        np.save(CONFIG_FILE, config)
        
        print(f"\n   💾 Конфигурация сохранена: {CONFIG_FILE}")
        print(f"      Антенн: {len(self.antennas)}")
        print(f"      Файл готов для использования в scan_real_3d.py")
    
    def _run_calculation(self):
        """Запускает расчет с текущей конфигурацией"""
        if not self.antennas:
            print("   ⚠️  Сначала добавьте хотя бы одну антенну!")
            return
        
        # Сначала сохраняем
        self._save_config()
        
        # Закрываем окно редактора
        self.plotter.close()
        
        print("\n🚀 Запуск расчета с пользовательской конфигурацией...")
        print("   (откроется новое окно с результатами)\n")
        
        # Запускаем scan_real_3d.py с флагом использования пользовательских антенн
        # Установим переменную окружения
        os.environ['USE_CUSTOM_ANTENNAS'] = '1'
        
        try:
            # Импортируем и запускаем (без subprocess для простоты)
            import scan_real_3d
            scan_real_3d.run_real_radiometry(use_custom_antennas=True)
        except Exception as e:
            print(f"   ❌ Ошибка при запуске расчета: {e}")
            import traceback
            traceback.print_exc()


def main():
    editor = AntennaEditor()
    editor.run()


if __name__ == "__main__":
    main()