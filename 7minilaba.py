# Импорт необходимых библиотек
import numpy as np  # Для работы с массивами и математическими операциями
import matplotlib.pyplot as plt  # Для визуализации данных
from sklearn import datasets  # Для генерации тестовых данных
from sklearn.model_selection import train_test_split  # Для разделения данных на обучающую и тестовую выборки
from sklearn.neighbors import KNeighborsClassifier  # Метод k-ближайших соседей
from sklearn.svm import SVC  # Метод опорных векторов
from sklearn.tree import DecisionTreeClassifier  # Дерево решений

# =============================================
# 1. ГЕНЕРАЦИЯ РАЗЛИЧНЫХ ТИПОВ ДАННЫХ ДЛЯ КЛАССИФИКАЦИИ
# =============================================

# Создаем список из 5 различных наборов данных
datasets_list = [
    # 1. Два концентрических круга с небольшим шумом
    datasets.make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=30),

    # 2. Два полумесяца (нелинейно разделимые данные)
    datasets.make_moons(n_samples=500, noise=0.05, random_state=30),

    # 3. Два кластера с разным разбросом точек
    datasets.make_blobs(n_samples=500, cluster_std=[1.0, 0.5], random_state=30, centers=2),

    # 4. Место для анизотропных данных (заполним ниже)
    None,

    # 5. Два слабо пересекающихся кластера
    datasets.make_blobs(n_samples=500, random_state=30, centers=2)
]

# Создаем анизотропные данные (4-й набор данных)
# Сначала генерируем обычные кластеры
x, y = datasets.make_blobs(n_samples=500, random_state=170, centers=2)
# Применяем линейное преобразование для создания анизотропии
transformation = [[0.6, -0.6], [-0.4, 0.8]]
x_aniso = np.dot(x, transformation)
# Заменяем None на наши анизотропные данные
datasets_list[3] = (x_aniso, y)

# Названия для каждого набора данных (для подписей графиков)
dataset_names = [
    "Концентрические круги",
    "Полумесяцы",
    "Кластеры разного размера",
    "Анизотропные данные",
    "Слабо пересекающиеся кластеры"
]

# =============================================
# 2. СОЗДАНИЕ И НАСТРОЙКА КЛАССИФИКАТОРОВ
# =============================================

# Инициализируем три разных классификатора:
classifiers = [
    # 1. Метод k-ближайших соседей (k=3)
    ("KNN", KNeighborsClassifier(n_neighbors=3)),

    # 2. Метод опорных векторов с RBF-ядром
    ("SVM", SVC(kernel='rbf', C=1.0)),

    # 3. Дерево решений с максимальной глубиной 5
    ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=42))
]

# =============================================
# 3. ПОДГОТОВКА ГРАФИКОВ ДЛЯ ВИЗУАЛИЗАЦИИ
# =============================================

# Создаем фигуру с 5 строками (по одной на каждый набор данных)
# и 3 столбцами (по одному на каждый классификатор)
fig, axes = plt.subplots(5, 3, figsize=(15, 20))

# Настраиваем отступы между графиками
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# =============================================
# 4. ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ДАННЫХ
# =============================================

# Перебираем все наборы данных
for i, (X, y) in enumerate(datasets_list):
    # Разделяем данные на обучающую (80%) и тестовую (20%) выборки
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Определяем границы для построения сетки
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Создаем сетку для построения границ решений
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Перебираем все классификаторы
    for j, (name, clf) in enumerate(classifiers):
        # Обучаем классификатор на обучающих данных
        clf.fit(X_train, y_train)

        # Прогнозируем классы для всех точек сетки
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Прогнозируем классы для тестовых данных
        y_pred = clf.predict(X_test)

        # =============================================
        # 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ
        # =============================================

        # Рисуем области решений (разные цвета для разных классов)
        axes[i, j].contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

        # Отображаем обучающие данные
        for k in range(len(X_train)):
            # Класс 0 - маркер 'x', класс 1 - маркер 'o'
            marker = 'x' if y_train[k] == 0 else 'o'
            axes[i, j].scatter(X_train[k, 0], X_train[k, 1],
                               marker=marker, c='blue', alpha=0.7, s=20)

        # Отображаем тестовые данные
        for k in range(len(X_test)):
            # Правильные классификации - зеленые, ошибки - красные
            color = 'green' if y_test[k] == y_pred[k] else 'red'
            marker = 'x' if y_test[k] == 0 else 'o'
            axes[i, j].scatter(X_test[k, 0], X_test[k, 1],
                               marker=marker, c=color, s=30)

        # Добавляем заголовок к графику
        axes[i, j].set_title(f"{dataset_names[i]}\n{name}", fontsize=10)

        # Устанавливаем границы осей
        axes[i, j].set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
        axes[i, j].set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

# =============================================
# 6. ОТОБРАЖЕНИЕ ВСЕХ ГРАФИКОВ
# =============================================

plt.show()



