import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
import random
from itertools import cycle

# 1. Выбор 3 алгоритмов кластеризации
selected_algorithms = [
    ('KMeans', KMeans(n_clusters=3)),
    ('DBSCAN', DBSCAN(eps=0.3)),
    ('Agglomerative', AgglomerativeClustering(n_clusters=3))
]


# 2. Генерация 6 различных наборов данных
def generate_datasets():
    np.random.seed(42)
    random.seed(42)

    # 1. Кластеры с разной дисперсией
    X1, y1 = make_blobs(n_samples=300, centers=3, cluster_std=[1.0, 2.5, 0.5], random_state=42)

    # 2. Полумесяцы
    X2, y2 = make_moons(n_samples=300, noise=0.05, random_state=42)

    # 3. Концентрические круги
    X3, y3 = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

    # 4. Анизотропные данные
    X4, y4 = make_blobs(n_samples=300, centers=3, random_state=42)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X4 = np.dot(X4, transformation)

    # 5. Кластеры с выбросами
    X5, y5 = make_blobs(n_samples=250, centers=3, cluster_std=1.0, random_state=42)
    X5 = np.concatenate([X5, np.random.uniform(low=-10, high=10, size=(50, 2))])
    y5 = np.concatenate([y5, np.full(50, -1)])

    # 6. Неравномерные кластеры
    X6, y6 = make_blobs(n_samples=300, centers=3, random_state=42)
    X6 = X6 * np.array([1, 4])

    datasets = [
        ('Кластеры с разной дисперсией', X1),
        ('Полумесяцы', X2),
        ('Концентрические круги', X3),
        ('Анизотропные данные', X4),
        ('Кластеры с выбросами', X5),
        ('Неравномерные кластеры', X6)
    ]

    # Масштабирование данных
    for i, (name, data) in enumerate(datasets):
        datasets[i] = (name, StandardScaler().fit_transform(data))

    return datasets


# 3. Функция для визуализации результатов
def plot_clusters(datasets, algorithm_results):
    plt.figure(figsize=(15, 20))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    colors = cycle('bgrcmyk')

    for i, (dataset_name, X) in enumerate(datasets):
        for j, (algo_name, _) in enumerate(selected_algorithms):
            plt.subplot(len(datasets), len(selected_algorithms), i * len(selected_algorithms) + j + 1)

            labels = algorithm_results[i][j]
            unique_labels = set(labels)
            current_colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

            for k, col in zip(unique_labels, current_colors):
                if k == -1:
                    col = (0, 0, 0, 1)  # Черный для шума

                class_member_mask = (labels == k)
                xy = X[class_member_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                         markeredgecolor='k', markersize=6 if k != -1 else 4)

            plt.title(f'{algo_name}\nна {dataset_name}')
            plt.xticks([])
            plt.yticks([])

    plt.show()


# 4. Основной код программы
datasets = generate_datasets()

# Применение алгоритмов
algorithm_results = []
for dataset_name, X in datasets:
    dataset_results = []
    for algo_name, algo in selected_algorithms:
        if algo_name == 'KMeans':
            algo.set_params(n_clusters=3)
        elif algo_name == 'DBSCAN':
            if dataset_name == 'Концентрические круги':
                algo.set_params(eps=0.2)
            elif dataset_name == 'Полумесяцы':
                algo.set_params(eps=0.3)
            else:
                algo.set_params(eps=0.5)

        labels = algo.fit_predict(X)
        dataset_results.append(labels)
    algorithm_results.append(dataset_results)

# Визуализация
plot_clusters(datasets, algorithm_results)

# Вывод таблицы результатов
print("\nРезультаты кластеризации:")
print("| Набор данных                 | KMeans | DBSCAN | Agglomerative |")
print("|------------------------------|--------|--------|---------------|")
for i, (dataset_name, _) in enumerate(datasets):
    row = f"| {dataset_name.ljust(28)} |"
    for j in range(len(selected_algorithms)):
        n_clusters = len(set(algorithm_results[i][j])) - (1 if -1 in algorithm_results[i][j] else 0)
        row += f" {str(n_clusters).center(6)} |"
    print(row)