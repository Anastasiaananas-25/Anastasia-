# Импорт необходимых библиотек
import numpy as np  # Для работы с массивами и математических операций
import matplotlib.pyplot as plt  # Для визуализации данных
from sklearn.linear_model import LinearRegression, Ridge  # Линейные модели регрессии
from sklearn.svm import SVR  # Метод опорных векторов для регрессии
from sklearn.tree import DecisionTreeRegressor  # Дерево решений для регрессии
from sklearn.ensemble import RandomForestRegressor  # Случайный лес для регрессии
from sklearn.metrics import mean_squared_error  # Метрика качества MSE
import random  # Для генерации случайных чисел


# 1. Определение исходной функции
def original_function(x):
    return np.sin(x * 2) + 0.5 * x + 0.1 * x ** 2 + 0.05 * x ** 3


# 2. Проверка типа функции (линейная/нелинейная)
# Наша функция явно нелинейна из-за наличия sin и полиномов высоких степеней
is_linear = False

# 3. Выбор методов регрессии в зависимости от типа функции
if is_linear:
    # Линейные методы регрессии (не будут использованы, так как функция нелинейна)
    methods = [
        ("Линейная регрессия", LinearRegression()),
        ("Ridge регрессия", Ridge(alpha=1.0)),
        ("Lasso регрессия", Lasso(alpha=0.1))
    ]
else:
    # Нелинейные методы регрессии (используются для нашей задачи)
    methods = [
        ("Метод опорных векторов (SVR)", SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)),
        ("Случайный лес", RandomForestRegressor(n_estimators=100, random_state=42)),
        ("Дерево решений", DecisionTreeRegressor(max_depth=5, random_state=42))
    ]

# 4. Генерация данных с шумом
np.random.seed(42)  # Фиксируем случайное зерно для воспроизводимости
x_min, x_max = -5, 5  # Диапазон значений по оси X

# Создаем 100 равномерно распределенных точек по X
x = np.linspace(x_min, x_max, 100).reshape(-1, 1)

# Вычисляем "чистые" значения Y (без шума)
y_true = original_function(x).ravel()

# Генерируем шум в диапазоне [-0.5, 0.5]
noise = np.random.uniform(-0.5, 0.5, size=len(x))

# Добавляем шум к исходным значениям Y
y_noisy = y_true + noise

# 5. Обучение моделей и получение предсказаний
results = []  # Список для хранения результатов
for name, model in methods:
    # Обучение модели на зашумленных данных
    model.fit(x, y_noisy)

    # Получение предсказаний для всех точек X
    y_pred = model.predict(x)

    # Вычисление среднеквадратичной ошибки (MSE)
    mse = mean_squared_error(y_true, y_pred)

    # Сохраняем результаты для последующей визуализации
    results.append((name, y_pred, mse))

# 6. Визуализация результатов
plt.figure(figsize=(15, 10))  # Создаем фигуру большого размера

# Строим отдельные графики для каждого метода
for i, (name, y_pred, mse) in enumerate(results, 1):
    plt.subplot(2, 2, i)  # Создаем подграфик (2 строки, 2 столбца)

    # Отображаем зашумленные данные синими точками
    plt.scatter(x, y_noisy, color='blue', label='Зашумленные данные', alpha=0.5)

    # Рисуем график исходной функции зеленой линией
    plt.plot(x, y_true, color='green', linewidth=2, label='Истинная функция')

    # Рисуем график предсказаний красной линией
    plt.plot(x, y_pred, color='red', linewidth=2, label=f'Предсказание ({name})')

    # Настройки графика
    plt.title(f"{name}\nMSE: {mse:.4f}")  # Заголовок с названием метода и MSE
    plt.xlabel('x')  # Подпись оси X
    plt.ylabel('y')  # Подпись оси Y
    plt.legend()  # Отображаем легенду
    plt.grid(True)  # Включаем сетку

plt.tight_layout()  # Автоматическая настройка отступов между графиками
plt.show()  # Отображаем все графики