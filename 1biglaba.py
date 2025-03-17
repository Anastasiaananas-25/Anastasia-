#использовать метод k ближайших соседей для реализации классификации точек в двумерном пространстве
import matplotlib.pyplot as plt
import random
x = []
y = []
pointsCount1 = 50
pointsCount2 = 50
class2_points = []
for i in range(50):
    x.append([random.uniform(1,8),random.uniform(1,8)])
    y.append(0)
class1_points = x
for i in range(50):
    class2_points.append([random.uniform(7,12),random.uniform(7,12)])
    y.append(1)
x = x + class2_points

def train_test_split(x,y,p=0.8):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    g = []
    for i in range(80):
        index = random.randint(0,len(x)-1)
        if index not in g:
            x_train.append(x[index])
            y_train.append(y[index])
            g.append(index)
            x.remove(x[index])
            y.remove(y[index])
    x_test = x
    y_test = y
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split(x, y)

# Реализация метода k ближайших соседей
def fit(x_train, y_train, x_test, k=3):
    y_predict = []
    for test_point in x_test:
        distances = []
        i = 0
        for train_point in x_train:
            distance = ((train_point[0] - test_point[0]) ** 2 + (train_point[1] - test_point[1]) ** 2)**(0.5)
            distances.append((distance, y_train[i]))
            i += 1
        
        # Сортировка по расстоянию
        distances.sort(key=lambda x: x[0])
        # Получение меток классов ближайших соседей
        neighbors = [distances[i][1] for i in range(k)]
        # Определение наиболее частого класса
        y_predict.append(max(set(neighbors), key=neighbors.count))
    
    return y_predict

# Реализация функции для расчета метрики accuracy
def computeAccuracy(y_test, y_predict):
    correct = sum(1 for yt, yp in zip(y_test, y_predict) if yt == yp)
    accuracy = correct / len(y_test)
    return accuracy

# Классификация точек из тестовой выборки
k = 3
y_predict = fit(x_train, y_train, x_test, k)

# Оценка точности работы алгоритма
accuracy = computeAccuracy(y_test, y_predict)
print(f'Accuracy: {accuracy * 100:.2f}%')

def visualize(x_train, y_train, x_test, y_test, y_predict):
    plt.figure(figsize=(10, 6))
    
    # Обучающие точки
    for i, point in enumerate(x_train):
        if y_train[i] == 0:
            plt.scatter(point[0], point[1], color='blue', marker='o', label='Class 0' if i == 0 else "")
        else:
            plt.scatter(point[0], point[1], color='blue', marker='x', label='Class 1' if i == 0 else "")
    
    # Тестовые точки
    for i, point in enumerate(x_test):
        if y_test[i] == y_predict[i]:  # Верно классифицированные
            if y_test[i] == 0:
                plt.scatter(point[0], point[1], color='green', marker='o')
            else:
                plt.scatter(point[0], point[1], color='green', marker='x')
        else:  # Неверно классифицированные
            if y_test[i] == 0:
                plt.scatter(point[0], point[1], color='red', marker='o')
            else:
                plt.scatter(point[0], point[1], color='red', marker='x')
    
    plt.title('K-Nearest Neighbors Classification')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid()
    plt.show()
visualize(x_train, y_train, x_test, y_test, y_predict)
    
    
