import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


# 1. Создаём 3 круга с точками
def generate_circle_data(center, radius, n_points):
    points = []
    for _ in range(n_points):
        corner = np.random.uniform(0, 2 * np.pi)
        r = radius * np.sqrt(np.random.uniform(0, 1))
        x = center[0] + r * np.cos(corner)
        y = center[1] + r * np.sin(corner)
        points.append([x, y])
    return np.array(points)


# Создаем 3 кластера
cluster1 = generate_circle_data([2, 6], 1.5, 35)
cluster2 = generate_circle_data([5, 2], 1.2, 35)
cluster3 = generate_circle_data([8, 7], 1.0, 35)
X = np.vstack([cluster1, cluster2, cluster3])


# 2. Реализация K-means
def k_means(X, k=3, max_iter=100):
    # Создаём случайные центроиды
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    history = [(centroids.copy(), None)]

    for _ in range(max_iter):
        # Вычислем расстояний от каждой точки до центроидов
        distances = np.sqrt(((X[:, np.newaxis] - centroids) ** 2).sum(axis=2))

        # Назначаем метки (индекс ближайшего центроида)
        labels = np.argmin(distances, axis=1)
        history.append((centroids.copy(), labels.copy())) #для создания ползунка

        # Проверяем условия остановки
        if len(history) > 2 and np.array_equal(history[-1][1], history[-2][1]):
            break

        # Пересчитываем центроиды
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        centroids = new_centroids

    return history


history = k_means(X, k=3)

# 3. Визуализация со слайдером
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.25)

# Начальное состояние
scatter = ax.scatter(X[:, 0], X[:, 1], c=history[1][1], cmap='viridis')
centers_scatter = ax.scatter(history[0][0][:, 0], history[0][0][:, 1],
                             c='red', marker='X', s=200)

# Слайдер
slider_ax = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(slider_ax, 'Итерация', 0, len(history) - 2, valinit=0, valfmt='%d')


# Функция обновления
def update(val):
    iteration = int(slider.val)
    centers, labels = history[iteration + 1]  # +1 потому что первая запись - инициализация
    scatter.set_array(labels)
    centers_scatter.set_offsets(centers)
    ax.set_title(f'K-means (Итерация {iteration}, Всего: {len(history) - 2})')
    fig.canvas.draw_idle()


slider.on_changed(update)
ax.set_title(f'K-means (Итерация 0, Всего: {len(history) - 2})')
plt.show()