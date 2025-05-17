import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random


# 1. Исходная функция (линейная)
def linear_function(x, k, b):
    """Линейная функция y = kx + b"""
    return k * x + b


# 2. Генерация данных с шумом
def generate_data(k_true, b_true, x_min=-10, x_max=10, points=100):
    """Генерация данных с шумом"""
    x = np.linspace(x_min, x_max, points)
    y_true = linear_function(x, k_true, b_true)
    noise = np.array([random.uniform(-3, 3) for _ in range(points)])
    y_noisy = y_true + noise
    return x, y_true, y_noisy


# 3. Функции для вычисления частных производных
def get_dk(x, y, k, b):
    """Частная производная по k"""
    return (2 / len(x)) * np.sum(x * (linear_function(x, k, b) - y))


def get_db(x, y, k, b):
    """Частная производная по b"""
    return (2 / len(x)) * np.sum(linear_function(x, k, b) - y)


# 4. Функция градиентного спуска
def gradient_descent(x, y, k_init, b_init, learning_rate=0.01, epochs=1000):
    """Реализация градиентного спуска"""
    k = k_init
    b = b_init
    k_history = [k]
    b_history = [b]

    for _ in range(epochs):
        # Вычисляем градиенты
        dk = get_dk(x, y, k, b)
        db = get_db(x, y, k, b)

        # Обновляем параметры
        k = k - learning_rate * dk
        b = b - learning_rate * db

        # Сохраняем историю параметров
        k_history.append(k)
        b_history.append(b)

    return k_history, b_history


# 5. Визуализация с ползунком
def interactive_plot(x, y_noisy, k_history, b_history):
    """Интерактивная визуализация процесса обучения"""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    # Исходные данные
    scatter = ax.scatter(x, y_noisy, color='blue', label='Зашумленные данные')
    line, = ax.plot(x, linear_function(x, k_history[0], b_history[0]),
                    'r-', linewidth=2, label='Регрессия')

    # Настройки графика
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Градиентный спуск для линейной регрессии')
    ax.grid(True)
    ax.legend()

    # Создание слайдера
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Эпоха', 0, len(k_history) - 1, valinit=0, valstep=1)

    # Функция обновления графика
    def update(val):
        epoch = int(slider.val)
        line.set_ydata(linear_function(x, k_history[epoch], b_history[epoch]))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


# Основная программа
if __name__ == "__main__":
    # Истинные параметры (из варианта 1: y = kx + b)
    k_true = 2.5
    b_true = -1.3

    # Генерация данных
    x, y_true, y_noisy = generate_data(k_true, b_true)

    # Инициализация параметров
    k_init = 0.0  # Начальное значение k
    b_init = 0.0  # Начальное значение b
    learning_rate = 0.01  # Скорость обучения
    epochs = 500  # Количество эпох

    # Градиентный спуск
    k_history, b_history = gradient_descent(x, y_noisy, k_init, b_init, learning_rate, epochs)

    # Вывод финальных параметров
    print(f"Истинные параметры: k = {k_true}, b = {b_true}")
    print(f"Найденные параметры: k = {k_history[-1]:.4f}, b = {b_history[-1]:.4f}")

    # Интерактивная визуализация
    interactive_plot(x, y_noisy, k_history, b_history)



