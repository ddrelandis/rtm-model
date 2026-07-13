import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import ndimage

# === 1. Путь к папке с DICOM ===
dicom_dir = r"/home/user/Downloads/40839/06682"  # замени на свой путь

# === 2. Загрузка ===
slices = []
for f in os.listdir(dicom_dir):
    if f.endswith('.dcm'):
        ds = pydicom.dcmread(os.path.join(dicom_dir, f))
        slices.append(ds)
slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
print(f"Загружено срезов: {len(slices)}")

# === 3. Объём ===
volume = np.stack([s.pixel_array.astype(float) for s in slices], axis=0)
if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
    volume = volume * float(slices[0].RescaleSlope) + float(slices[0].RescaleIntercept)

# === 4. Маска тела ===
body_mask_3d = np.zeros_like(volume, dtype=bool)
for i in range(volume.shape[0]):
    slice_mask = volume[i] > 70
    slice_mask = ndimage.binary_fill_holes(slice_mask)
    labeled, num = ndimage.label(slice_mask)
    if num > 0:
        sizes = ndimage.sum(slice_mask, labeled, range(1, num+1))
        largest = np.argmax(sizes) + 1
        body_mask_3d[i] = (labeled == largest)

body_voxels = volume[body_mask_3d]
body_voxels_clean = body_voxels[body_voxels > 0]
print(f"Вокселей для GMM: {len(body_voxels_clean)}")

# === 5. GMM с сортировкой компонентов по интенсивности ===
gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full')
gmm.fit(body_voxels_clean.reshape(-1, 1))

# Сортируем компоненты по среднему значению (от тёмного к яркому)
order = np.argsort(gmm.means_.flatten())
gmm.means_ = gmm.means_[order]
gmm.covariances_ = gmm.covariances_[order]
gmm.weights_ = gmm.weights_[order]
gmm.precisions_ = gmm.precisions_[order]
gmm.precisions_cholesky_ = gmm.precisions_cholesky_[order]

# Предсказываем метки ТОЛЬКО для ненулевых вокселей
labels_clean = gmm.predict(body_voxels_clean.reshape(-1, 1))

# Для 3D-сегментации: создаём полную маску с метками
segmentation_3d = np.zeros_like(volume, dtype=int)
body_indices = np.where(body_mask_3d)
nonzero_in_body = body_voxels > 0
segmentation_3d[body_mask_3d & (volume > 0)] = labels_clean + 1

# === 6. Визуализация ===
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
mid_idx = volume.shape[0] // 2

# Оригинальный срез
axes[0, 0].imshow(volume[mid_idx], cmap='gray')
axes[0, 0].set_title("Оригинал (z=50)")
axes[0, 0].axis('off')

# Маска тела
axes[0, 1].imshow(body_mask_3d[mid_idx], cmap='gray')
axes[0, 1].set_title("Маска тела")
axes[0, 1].axis('off')

# Сегментация
cmap = plt.cm.get_cmap('nipy_spectral', 4)
axes[0, 2].imshow(segmentation_3d[mid_idx], cmap=cmap, vmin=0, vmax=3)
axes[0, 2].set_title("GMM сегментация (3 класса)")
axes[0, 2].axis('off')

# Гистограмма с компонентами GMM
x_range = np.linspace(body_voxels_clean.min(), body_voxels_clean.max(), 300).reshape(-1, 1)
log_prob = gmm.score_samples(x_range)
pdf = np.exp(log_prob)

axes[1, 0].hist(body_voxels_clean, bins=150, density=True, alpha=0.4, color='coral', label='Данные')
axes[1, 0].plot(x_range, pdf, 'b-', linewidth=2, label='GMM fit')

colors = ['red', 'green', 'blue']
for i in range(3):
    mean = gmm.means_[i, 0]
    std = np.sqrt(gmm.covariances_[i, 0, 0])
    weight = gmm.weights_[i]
    x_comp = np.linspace(mean - 3*std, mean + 3*std, 100)
    y_comp = weight * np.exp(-0.5 * ((x_comp - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
    axes[1, 0].plot(x_comp, y_comp, color=colors[i], linewidth=2, label=f'Класс {i+1} (μ={mean:.0f})')

axes[1, 0].set_title("GMM компоненты")
axes[1, 0].set_xlabel("Интенсивность")
axes[1, 0].set_ylabel("Плотность")
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)

# Распределение классов
unique, counts = np.unique(labels_clean, return_counts=True)
axes[1, 1].bar(unique + 1, counts, color=[colors[i] for i in unique], alpha=0.7)
axes[1, 1].set_title("Распределение классов")
axes[1, 1].set_xlabel("Класс")
axes[1, 1].set_ylabel("Количество вокселей")
axes[1, 1].set_xticks([1, 2, 3])

# Средняя интенсивность по классам (теперь размеры совпадают)
class_means = [body_voxels_clean[labels_clean == i].mean() for i in range(3)]
axes[1, 2].bar([1, 2, 3], class_means, color=colors, alpha=0.7)
axes[1, 2].set_title("Средняя интенсивность по классам")
axes[1, 2].set_xlabel("Класс")
axes[1, 2].set_ylabel("Интенсивность")
axes[1, 2].set_xticks([1, 2, 3])

plt.tight_layout()
plt.show()

# === 7. Статистика ===
print("\n=== Статистика по классам GMM ===")
tissue_names = ['Тёмные ткани (фон/мышцы/строма)', 'Средние ткани (паренхима)', 'Яркие структуры (контраст/сосуды)']
for i in range(3):
    class_voxels = body_voxels_clean[labels_clean == i]
    pct = len(class_voxels) / len(body_voxels_clean) * 100
    print(f"Класс {i+1} — {tissue_names[i]}:")
    print(f"  Вокселей: {len(class_voxels)} ({pct:.1f}%)")
    print(f"  Среднее: {class_voxels.mean():.1f}, медиана: {np.median(class_voxels):.1f}")
    print(f"  Диапазон: [{class_voxels.min():.1f}, {class_voxels.max():.1f}]")