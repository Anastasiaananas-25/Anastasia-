import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.cluster import KMeans

#Создаём три области круга с точками
circles = [[[2, 6], 1.5],[[5, 2], 1.2],[[8, 7], 1.0]]

# Заполняем точками, 35 точек внутри каждого круга
points = []
for (x, y), r in circles:
    for _ in range(35):
        corner = np.random.uniform(0, 2 * np.pi) #случайный угол от 0 до 2pi
        radius = r * np.sqrt(np.random.uniform(0, 1)) # корень, чтобы точки были более-менее равномерно распределены
        points.append([x + radius * np.cos(corner), y + radius * np.sin(corner)])

X = np.array(points)
def run_kmeans(X, n_clusters=3, max_iter=100):
    history = [] #для создания ползунка

    # Инициализация
    kmeans = KMeans(n_clusters=n_clusters, init='random', max_iter=1, n_init=1)
    kmeans.fit(X)
    history.append((kmeans.cluster_centers_.copy(), kmeans.labels_.copy()))

    # Итерации
    for _ in range(max_iter):
        prev_labels = kmeans.labels_.copy()
        kmeans.max_iter += 1
        kmeans.fit(X)
        history.append((kmeans.cluster_centers_.copy(), kmeans.labels_.copy()))
        if np.array_equal(prev_labels, kmeans.labels_):
            break # условие для прекращения перемещений центридов

    return history


history = run_kmeans(X, n_clusters=3, max_iter=100)

# 3. Создание графика со слайдером
fig, ax = plt.subplots(figsize=(10, 7))
plt.subplots_adjust(bottom=0.25)

# Начальный кадр
scatter = ax.scatter(X[:, 0], X[:, 1], c=history[0][1], cmap='rainbow')
centers_scatter = ax.scatter(history[0][0][:, 0], history[0][0][:, 1],
                             c='black', marker='X', s=100)

# Слайдер
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Итерация', 0, len(history) - 1, valinit=0, valfmt='%d')


# Обновление графика
def update(val):
    iteration = int(slider.val)
    centers, labels = history[iteration]
    scatter.set_array(labels)
    centers_scatter.set_offsets(centers)
    fig.canvas.draw_idle()


slider.on_changed(update)

plt.title(f'K-means кластеризация')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()