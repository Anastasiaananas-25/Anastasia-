import numpy as np
import matplotlib.pyplot as plt


def gradientDescend(func=lambda x: x ** 2, diffFunc=lambda x: 2 * x, x0=3, speed=0.01, epochs=100):
    """
    Реализация метода градиентного спуска

    Параметры:
    func - функция, минимум которой ищем
    diffFunc - производная функции
    x0 - начальная точка
    speed - скорость обучения
    epochs - количество итераций

    Возвращает:
    x_list - список значений x на каждой итерации
    y_list - список значений функции в этих точках
    """
    x_list = []
    y_list = []
    x = x0

    for _ in range(epochs):
        x_list.append(x)
        y_list.append(func(x))

        # Обновление x по формуле градиентного спуска
        x = x - speed * diffFunc(x)

    return x_list, y_list


# 1. Генерируем функцию и её производную
# Используем функцию f(x) = sin(x) + 0.1*x^2, которая имеет несколько минимумов
def func(x):
    return np.sin(x) + 0.1 * x ** 2


def diffFunc(x):
    return np.cos(x) + 0.2 * x


# 2. Применяем градиентный спуск
x_list, y_list = gradientDescend(func, diffFunc, x0=3, speed=0.1, epochs=50)

# 3. Строим график
x_vals = np.linspace(-5, 5, 400)
y_vals = func(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x) = sin(x) + 0.1x²')
plt.scatter(x_list, y_list, color='red', label='Точки градиентного спуска')
plt.scatter(x_list[-1], y_list[-1], color='green', s=100, label='Найденный минимум')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Градиентный спуск для функции f(x) = sin(x) + 0.1x²')
plt.legend()
plt.grid(True)
plt.show()

# 4. Определяем, сходится ли метод к минимуму
print(f"Начальная точка: {x_list[0]}")
print(f"Найденная точка минимума: {x_list[-1]:.4f}")
print(f"Значение функции в этой точке: {y_list[-1]:.4f}")


# 5. Находим граничное значение speed
def find_critical_speed(func, diffFunc, x0=3, epochs=50, tol=1e-2):
    """
    Находит критическое значение скорости обучения,
    при котором метод перестает сходиться
    """
    low = 0.01
    high = 1.0

    while high - low > tol:
        mid = (high + low) / 2
        x_list, _ = gradientDescend(func, diffFunc, x0, mid, epochs)

        # Проверяем, сходится ли метод (значение x не уходит в бесконечность)
        if abs(x_list[-1]) > 100 or np.isnan(x_list[-1]):
            high = mid
        else:
            low = mid

    return (high + low) / 2


critical_speed = find_critical_speed(func, diffFunc)
print(f"\nГраничное значение speed: {critical_speed:.4f}")



